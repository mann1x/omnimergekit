#!/usr/bin/env python3
"""Extract per-source task-success competence maps from solved eval samples.

For each PASSING (sample) in lm_eval/lcb output JSONL:
  1. Tokenize prompt + the model's own correct completion
  2. Forward + per-token CE loss on completion tokens
  3. Backward
  4. Accumulate four importance signals per parameter tensor:
       - <name>           = Σ |W.grad|             (primary, Fisher-compat with omnimergekit)
       - <name>.grad_sq   = Σ (W.grad)^2           (Fisher diagonal)
       - <name>.weight_taylor = Σ |W * W.grad|     (TaylorFO at weight level)
       - <name>.act_taylor    = Σ |x * ∂L/∂x|      (TaylorFO at input-channel level, 1D)

Output: one safetensors per source. The primary key (= name) is shape-compatible
with omnimergekit's --fisher consumer: drop it in directly, or feed all four
signals to competence_combine.py and pick a blend.

Usage:
  python competence_extract.py \
      --model /path/to/source_hf_model \
      --samples /path/to/eval_results/humaneval_<src>/<src>/samples_*.jsonl \
      --task he \
      --output competence/<src>__he.safetensors \
      --max-samples 80 --max-len 1024
"""
import argparse
import gc
import glob
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import save_file


# ── Structured aggregation patterns ──────────────────────────────────────────
# Match attention/FFN linear weights across Llama/Qwen/Gemma/Mistral families
# (including MoE expert layers). Used by the --structured export path to fold
# per-element importance into per-head / per-neuron compact 1D signals.
_RE_ATTN_QKV = re.compile(r"\.self_attn\.(q|k|v)_proj\.weight$")
_RE_ATTN_O = re.compile(r"\.self_attn\.o_proj\.weight$")
_RE_FFN_GATE_UP = re.compile(r"\.(?:mlp|mlp\.experts\.\d+)\.(gate|up)_proj\.weight$")
_RE_FFN_DOWN = re.compile(r"\.(?:mlp|mlp\.experts\.\d+)\.down_proj\.weight$")


def _structured_aggregate(name: str, t: torch.Tensor,
                          cfg: Dict[str, int]) -> Optional[Tuple[str, torch.Tensor]]:
    """Reduce a full-shape per-element accumulator to a compact 1D structured signal.

    Returns (signal_class, compact_tensor) where signal_class is "head" or "neuron",
    or None if the tensor name doesn't match any structured pattern (e.g., layernorm,
    embed, lm_head). compact_tensor is fp32 on CPU.

    Naming convention: caller appends `.{signal_class}_compact_<accumulator>` suffix.
    """
    nh = cfg["num_heads"]
    nkv = cfg["num_kv_heads"]
    hd = cfg["head_dim"]
    inter = cfg["intermediate_size"]
    t = t.float()

    m = _RE_ATTN_QKV.search(name)
    if m is not None:
        which = m.group(1)
        heads = nh if which == "q" else nkv
        # [heads*head_dim, hidden] → [heads, head_dim, hidden] → sum over (head_dim, hidden)
        if t.shape[0] != heads * hd:
            return None  # unexpected shape (e.g. tied weights or fused projection)
        return ("head", t.reshape(heads, hd, t.shape[1]).sum(dim=(1, 2)))

    if _RE_ATTN_O.search(name) is not None:
        # [hidden, heads*head_dim] → [hidden, heads, head_dim] → sum over (hidden, head_dim)
        if t.shape[1] != nh * hd:
            return None
        return ("head", t.reshape(t.shape[0], nh, hd).sum(dim=(0, 2)))

    if _RE_FFN_GATE_UP.search(name) is not None:
        # [intermediate, hidden] → sum over hidden
        if t.shape[0] != inter:
            return None
        return ("neuron", t.sum(dim=1))

    if _RE_FFN_DOWN.search(name) is not None:
        # [hidden, intermediate] → sum over hidden
        if t.shape[1] != inter:
            return None
        return ("neuron", t.sum(dim=0))

    return None


SIDECAR_VERSION = 1


def _compute_config_hash(model_path: str, samples_path: str, max_samples: int,
                         max_len: int, chunk_len: int, skip_grad_patterns: str,
                         with_act_taylor: bool, task: Optional[str],
                         prompt_key: str, completion_key: str,
                         pass_key: str, pass_value: Any,
                         keep_doc_ids: Optional[set]) -> str:
    """Hash the inputs that materially shape the accumulators. Resume refuses on mismatch.

    Sample order in the JSONL drives accumulator state, so identical inputs MUST hash
    identically — order the keys and string-cast values for stable digest.
    """
    h = hashlib.sha256()
    parts = [
        ("model_path", model_path),
        ("samples_path", samples_path),
        ("max_samples", str(max_samples)),
        ("max_len", str(max_len)),
        ("chunk_len", str(chunk_len)),
        ("skip_grad_patterns", skip_grad_patterns or ""),
        ("with_act_taylor", "1" if with_act_taylor else "0"),
        ("task", task or "custom"),
        ("prompt_key", prompt_key),
        ("completion_key", completion_key),
        ("pass_key", pass_key),
        ("pass_value", str(pass_value)),
        ("keep_doc_ids", ",".join(sorted(keep_doc_ids)) if keep_doc_ids else ""),
    ]
    for k, v in parts:
        h.update(k.encode())
        h.update(b"\x00")
        h.update(v.encode())
        h.update(b"\x01")
    return h.hexdigest()


def _sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".ckpt.pt")


def _save_sidecar(sidecar: Path, extractor: "HookedExtractor", next_idx: int,
                  passing_count: int, config_hash: str, t0_wall: float,
                  ckpt_dtype: torch.dtype) -> None:
    """Atomic-write the sidecar via temp file + rename so OOM mid-write can't corrupt it."""
    state = {
        "version": SIDECAR_VERSION,
        "config_hash": config_hash,
        "n_samples_done": extractor._n_samples,
        "next_sample_idx": next_idx,
        "passing_count": passing_count,
        "acc_grad_l1": {k: v.to(ckpt_dtype) for k, v in extractor.acc_grad_l1.items()},
        "acc_grad_sq": {k: v.to(ckpt_dtype) for k, v in extractor.acc_grad_sq.items()},
        "acc_weight_taylor": {k: v.to(ckpt_dtype) for k, v in extractor.acc_weight_taylor.items()},
        "acc_act_taylor": {k: v.to(ckpt_dtype) for k, v in extractor.acc_act_taylor.items()},
        "wall_time_sec": time.time() - t0_wall,
        "saved_at": time.time(),
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    torch.save(state, str(tmp))
    os.replace(str(tmp), str(sidecar))


def _load_sidecar_or_refuse(sidecar: Path, config_hash: str,
                            passing_count: int) -> Optional[Dict[str, Any]]:
    """Load sidecar if compatible. Returns dict or None.

    None means: no sidecar OR mismatched config (caller must decide whether to refuse).
    Caller distinguishes the cases by checking sidecar.exists() before calling.
    """
    if not sidecar.exists():
        return None
    try:
        state = torch.load(str(sidecar), map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  WARN: sidecar at {sidecar} unreadable ({type(e).__name__}: {e}); ignoring",
              file=sys.stderr)
        return None
    if state.get("version") != SIDECAR_VERSION:
        print(f"  WARN: sidecar version {state.get('version')} != {SIDECAR_VERSION}; ignoring",
              file=sys.stderr)
        return None
    if state.get("config_hash") != config_hash:
        print("  REFUSE-RESUME: sidecar config_hash mismatch", file=sys.stderr)
        print(f"    sidecar : {state.get('config_hash')}", file=sys.stderr)
        print(f"    current : {config_hash}", file=sys.stderr)
        print(f"    Delete {sidecar} or restart without --resume.", file=sys.stderr)
        return "REFUSE"  # type: ignore  # signal to caller via sentinel
    if state.get("passing_count") != passing_count:
        print(f"  REFUSE-RESUME: passing_count mismatch "
              f"(sidecar={state.get('passing_count')} current={passing_count})",
              file=sys.stderr)
        return "REFUSE"  # type: ignore
    return state


TASK_PRESETS = {
    "he": dict(
        prompt_key="arguments.gen_args_0.arg_0",
        completion_key="filtered_resps.0",
        pass_key="pass@1",
        pass_value=1.0,
    ),
    "mbpp": dict(
        prompt_key="arguments.gen_args_0.arg_0",
        completion_key="filtered_resps.0",
        pass_key="pass_at_1",
        pass_value=1.0,
    ),
    "lcb": dict(
        prompt_key="prompt",
        completion_key="completion",
        pass_key="passed",
        pass_value=True,
    ),
    "aime": dict(
        prompt_key="arguments.gen_args_0.arg_0",
        completion_key="filtered_resps.0",
        pass_key="exact_match",
        pass_value=1.0,
    ),
}


def get_dotted(obj: Any, path: str) -> Any:
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = obj[part]
    return obj


def load_passing(samples_path: Path, prompt_key: str, completion_key: str,
                 pass_key: str, pass_value: Any, max_samples: int,
                 keep_doc_ids: Optional[set] = None) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    n_doc_filter_drop = 0
    with open(samples_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if keep_doc_ids is not None:
                    did = str(rec.get("doc_id", ""))
                    if did not in keep_doc_ids:
                        n_doc_filter_drop += 1
                        continue
                p = get_dotted(rec, pass_key)
                if isinstance(pass_value, float):
                    if not (float(p) >= pass_value):
                        continue
                else:
                    if p != pass_value:
                        continue
                prompt = get_dotted(rec, prompt_key)
                comp = get_dotted(rec, completion_key)
                if isinstance(prompt, list):
                    prompt = prompt[0]
                if isinstance(comp, list):
                    comp = comp[0] if comp else ""
                if not prompt or not comp:
                    continue
                out.append((str(prompt), str(comp)))
                if len(out) >= max_samples:
                    break
            except (KeyError, IndexError, TypeError, ValueError) as e:
                print(f"  skip malformed: {e}", file=sys.stderr)
    if keep_doc_ids is not None:
        print(f"  doc_id filter: kept {len(out)}, dropped {n_doc_filter_drop} non-matching")
    return out


class HookedExtractor:
    """Owns per-Linear forward+input-grad hooks, accumulates per-tensor signals on CPU fp32."""

    def __init__(self, model: nn.Module, device: torch.device, with_act_taylor: bool = False):
        self.model = model
        self.device = device
        self.acc_grad_l1: Dict[str, torch.Tensor] = {}      # CPU fp32, shape = W
        self.acc_grad_sq: Dict[str, torch.Tensor] = {}      # CPU fp32, shape = W
        self.acc_weight_taylor: Dict[str, torch.Tensor] = {}  # CPU fp32, shape = W
        self.acc_act_taylor: Dict[str, torch.Tensor] = {}   # CPU fp32, shape = (in,)
        self._fwd_handles = []
        self._n_samples = 0
        self.with_act_taylor = with_act_taylor

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.acc_grad_l1[name] = torch.zeros(p.shape, dtype=torch.float32)
                self.acc_grad_sq[name] = torch.zeros(p.shape, dtype=torch.float32)
                self.acc_weight_taylor[name] = torch.zeros(p.shape, dtype=torch.float32)

        # Optional: input-activation × ∂L/∂x channel hook. Conflicts with gradient
        # checkpointing (captured tensor freed during backward), so keep it off
        # unless explicitly enabled and checkpointing disabled.
        if not with_act_taylor:
            return
        for mod_name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                def make_hook(name=mod_name):
                    def hook(module, inputs):
                        x = inputs[0]
                        if not x.requires_grad:
                            x.requires_grad_(True)
                        def grad_hook(g, captured_x=x, mname=name):
                            try:
                                contrib = (captured_x.detach() * g).abs()
                                contrib = contrib.reshape(-1, contrib.shape[-1]).sum(dim=0).float().cpu()
                                if mname not in self.acc_act_taylor:
                                    self.acc_act_taylor[mname] = contrib
                                else:
                                    self.acc_act_taylor[mname] += contrib
                            except Exception:
                                pass
                        x.register_hook(grad_hook)
                        return None
                    return hook
                self._fwd_handles.append(mod.register_forward_pre_hook(make_hook()))

    def flush_grads_to_cpu(self) -> None:
        """Copy any current .grad to CPU accumulators, then zero. Does NOT bump sample counter.
        Call between sequence chunks to prevent grad-buffer + activation peak from compounding."""
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            # weight·grad before transferring (cheaper as bf16 mul on GPU)
            wg = (p.detach() * g).abs().float().cpu()
            ga = g.abs().float().cpu()
            gs = (g.float() ** 2).cpu()
            self.acc_grad_l1[name] += ga
            self.acc_grad_sq[name] += gs
            self.acc_weight_taylor[name] += wg
            p.grad = None

    def mark_sample_done(self) -> None:
        """Bump the sample counter — call once per ORIGINAL sample (not per chunk)."""
        self._n_samples += 1

    def accumulate_weight_signals(self) -> None:
        """Legacy single-shot path: flush grads + bump counter. Equivalent to one chunk per sample."""
        self.flush_grads_to_cpu()
        self.mark_sample_done()

    def remove(self) -> None:
        for h in self._fwd_handles:
            h.remove()
        self._fwd_handles.clear()

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore accumulators from a sidecar dict (see _save_sidecar).

        Caller is responsible for config-compatibility checks. We copy in fp32 to
        match the in-memory accumulator dtype, regardless of the sidecar's dtype.
        Missing keys are tolerated (e.g., act_taylor when --with-act-taylor was off);
        extra keys in the sidecar that aren't in the current model are warned-and-skipped.
        """
        def _restore(target: Dict[str, torch.Tensor], src: Dict[str, torch.Tensor]) -> None:
            extra = set(src.keys()) - set(target.keys())
            if extra:
                print(f"  WARN: sidecar has {len(extra)} keys not in current model — skipping",
                      file=sys.stderr)
            for k in target.keys():
                if k in src:
                    target[k] = src[k].to(torch.float32)

        _restore(self.acc_grad_l1, state.get("acc_grad_l1") or {})
        _restore(self.acc_grad_sq, state.get("acc_grad_sq") or {})
        _restore(self.acc_weight_taylor, state.get("acc_weight_taylor") or {})
        # act_taylor is populated lazily; restore directly without target-key gating
        for k, v in (state.get("acc_act_taylor") or {}).items():
            self.acc_act_taylor[k] = v.to(torch.float32)
        self._n_samples = int(state.get("n_samples_done", 0))

    def export(self, out_dtype: torch.dtype = torch.float16,
               structured_config: Optional[Dict[str, int]] = None) -> Dict[str, torch.Tensor]:
        """Export per-element signals; if structured_config is given, also emit
        per-head / per-neuron compact 1D aggregates with `.head_compact_*` /
        `.neuron_compact_*` suffixes.
        """
        out: Dict[str, torch.Tensor] = {}
        n = max(self._n_samples, 1)
        for name, t in self.acc_grad_l1.items():
            out[name] = (t / n).to(out_dtype)
        for name, t in self.acc_grad_sq.items():
            out[f"{name}.grad_sq"] = (t / n).to(out_dtype)
        for name, t in self.acc_weight_taylor.items():
            out[f"{name}.weight_taylor"] = (t / n).to(out_dtype)
        for name, t in self.acc_act_taylor.items():
            out[f"{name}.act_taylor"] = (t / n).to(out_dtype)

        if structured_config is not None:
            # Aggregate each accumulator into compact head/neuron 1D signals.
            # Iterate by accumulator-suffix: primary (no suffix), grad_sq, weight_taylor.
            # act_taylor is already 1D (per-input-channel) and not structurally aggregated.
            sources = [
                ("l1", self.acc_grad_l1),
                ("sq", self.acc_grad_sq),
                ("taylor", self.acc_weight_taylor),
            ]
            n_head = n_neuron = 0
            for suffix_tag, acc in sources:
                for name, t in acc.items():
                    res = _structured_aggregate(name, t / n, structured_config)
                    if res is None:
                        continue
                    sig_class, compact = res
                    out[f"{name}.{sig_class}_compact_{suffix_tag}"] = compact.to(out_dtype)
                    if suffix_tag == "l1":  # count once per tensor
                        if sig_class == "head":
                            n_head += 1
                        else:
                            n_neuron += 1
            print(f"  structured: {n_head} head-aggregated tensors, "
                  f"{n_neuron} neuron-aggregated tensors")
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path,
                    help="HF model directory (the source fine-tune to extract competence from).")
    ap.add_argument("--samples", required=True, type=Path,
                    help="Path to lm_eval / lcb samples_*.jsonl. Glob ok if it resolves to a single file.")
    ap.add_argument("--task", choices=list(TASK_PRESETS), default=None,
                    help="Use schema preset for he/mbpp/lcb. If omitted, supply --prompt-key / --completion-key / --pass-key manually.")
    ap.add_argument("--prompt-key", default=None,
                    help="Dotted path to prompt in each sample (overrides preset).")
    ap.add_argument("--completion-key", default=None,
                    help="Dotted path to completion-to-learn (overrides preset).")
    ap.add_argument("--pass-key", default=None,
                    help="Dotted path to pass indicator (overrides preset).")
    ap.add_argument("--pass-value", default=None,
                    help="Pass-key value indicating success (overrides preset; parsed as float if numeric, else literal).")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output safetensors path.")
    ap.add_argument("--max-samples", type=int, default=80,
                    help="Cap solved samples used per task. Default 80 — most HE has ~100 passing.")
    ap.add_argument("--max-len", type=int, default=512,
                    help="Max combined token length per sample. Default 512 — fits most HE/MBPP after right-truncation. "
                         "Lower if OOM, raise if you have VRAM headroom. With --chunk-len, this is a hard ceiling on the "
                         "total sequence kept (still useful to bound 32k AIME completions).")
    ap.add_argument("--chunk-len", type=int, default=0,
                    help="If > 0, split each sample into non-overlapping windows of this size and run forward+backward "
                         "per chunk, accumulating gradients in-place. Lets long sequences fit on small VRAM by trading "
                         "peak activation memory for sequential compute. Approximation: each chunk is processed as an "
                         "independent sequence (no cross-chunk attention/state) — fine for fisher importance signal but "
                         "not for training. Default 0 = single forward+backward (legacy behavior).")
    ap.add_argument("--device", default=None,
                    help="cuda or cpu (auto-detect by default).")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                    help="Model load dtype. bf16 halves VRAM but limits some grad precision; matches eval setup.")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="Disable gradient checkpointing (faster but more VRAM).")
    ap.add_argument("--with-act-taylor", action="store_true",
                    help="Also capture input-activation × ∂L/∂x channel hook (1D per Linear). "
                         "Conflicts with gradient checkpointing — auto-disables checkpointing if set. "
                         "Needs ~30%% more VRAM. Off by default since the weight-level signals already "
                         "encode activation×grad via |W.grad| = |Σ grad_y * x|; the act_taylor 1D channel "
                         "view is mainly useful for downstream head/channel pruning.")
    ap.add_argument("--skip-grad-patterns", type=str,
                    default="embed_tokens,lm_head",
                    help="Comma-sep substrings; any param whose name contains ANY of these has requires_grad=False. "
                         "Default: 'embed_tokens,lm_head' — they're huge and skipped at merge anyway via "
                         "--pr682-turbo. Pass empty string to track everything.")
    ap.add_argument("--meta", type=str, default="",
                    help="Free-text metadata to embed in the safetensors file metadata.")
    ap.add_argument("--keep-doc-ids", type=str, default=None,
                    help="Comma-separated doc_id list. If set, ONLY samples whose 'doc_id' field matches "
                         "(after string-cast) are considered. Use to restrict extraction to a differential "
                         "set (e.g., problems THIS source uniquely solved vs another source). Combined with "
                         "the pass filter — both must hold.")
    ap.add_argument("--save-dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                    help="Output safetensors dtype. fp16 halves disk vs fp32 with negligible loss for "
                         "use as merge importance signal. fp32 only if you need exact accumulator values.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from a sidecar checkpoint at <output>.ckpt.pt if it exists and the "
                         "config_hash matches. Without this flag, an existing sidecar is renamed to "
                         "<output>.ckpt.pt.stale rather than silently overwritten.")
    ap.add_argument("--ckpt-every", type=int, default=10,
                    help="Save sidecar every N completed samples. Default 10. Set 0 to disable "
                         "sample-cadence checkpointing (still respects --ckpt-time-sec).")
    ap.add_argument("--ckpt-time-sec", type=int, default=300,
                    help="Save sidecar at least every N seconds (whichever fires first vs --ckpt-every). "
                         "Default 300. Set 0 to disable time-cadence checkpointing.")
    ap.add_argument("--ckpt-dtype", default="float16", choices=["float16", "float32"],
                    help="Sidecar accumulator dtype. fp16 halves disk; fp32 if you need exact restore "
                         "(rarely matters since we re-normalize in combine).")
    ap.add_argument("--structured", action="store_true",
                    help="At export time, additionally emit compact 1D per-head and per-neuron "
                         "aggregates of each weight-level accumulator. Adds keys with suffixes "
                         ".head_compact_l1/.head_compact_sq/.head_compact_taylor for q/k/v/o_proj "
                         "and .neuron_compact_l1/.neuron_compact_sq/.neuron_compact_taylor for "
                         "gate/up/down_proj (incl. MoE expert layers). The per-element tensors are "
                         "still saved as before; structured signals are additive. Embeds a "
                         "'structured_config' JSON in the safetensors file metadata so consumers "
                         "(combine, omnimergekit) can reshape correctly without re-loading the model.")
    args = ap.parse_args()

    # Resolve preset / overrides
    if args.task:
        preset = TASK_PRESETS[args.task].copy()
    else:
        preset = {}
    if args.prompt_key:
        preset["prompt_key"] = args.prompt_key
    if args.completion_key:
        preset["completion_key"] = args.completion_key
    if args.pass_key:
        preset["pass_key"] = args.pass_key
    if args.pass_value is not None:
        try:
            preset["pass_value"] = float(args.pass_value)
        except ValueError:
            preset["pass_value"] = args.pass_value
    for need in ("prompt_key", "completion_key", "pass_key", "pass_value"):
        if need not in preset:
            print(f"ERROR: missing {need} (pass --task or --{need.replace('_','-')})", file=sys.stderr)
            sys.exit(1)

    # Resolve samples (allow glob)
    samples_paths = sorted(glob.glob(str(args.samples)))
    if not samples_paths:
        print(f"ERROR: no samples file at {args.samples}", file=sys.stderr)
        sys.exit(1)
    if len(samples_paths) > 1:
        print(f"WARN: glob matched {len(samples_paths)} files, using {samples_paths[0]}", file=sys.stderr)
    samples_file = Path(samples_paths[0])
    print(f"  samples : {samples_file}")
    print(f"  preset  : {preset}")

    keep_ids = None
    if args.keep_doc_ids:
        keep_ids = set(s.strip() for s in args.keep_doc_ids.split(",") if s.strip())
        print(f"  keep-doc-ids: {len(keep_ids)} ids")

    # Load passing samples
    passing = load_passing(samples_file, preset["prompt_key"], preset["completion_key"],
                           preset["pass_key"], preset["pass_value"], args.max_samples,
                           keep_doc_ids=keep_ids)
    print(f"  passing : {len(passing)} samples (cap {args.max_samples})")
    if not passing:
        print("ERROR: no passing samples found — check pass_key / pass_value / completion_key", file=sys.stderr)
        sys.exit(1)

    # Device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # Load model + tokenizer (lightseek venv has transformers 5.5+)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  loading model {args.model} dtype={args.dtype} device={device}")
    tok = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="eager",  # required for grad through attention
    ).to(device)
    model.eval()  # disable dropout, but we still call backward — that's fine

    # act_taylor and gradient_checkpointing don't co-exist (captured x freed before backward fires)
    use_checkpoint = (not args.no_checkpoint) and (not args.with_act_taylor)
    if use_checkpoint:
        try:
            model.gradient_checkpointing_enable()
            print("  gradient_checkpointing: ON")
        except Exception as e:
            print(f"  gradient_checkpointing failed: {e} — continuing without")
    elif args.with_act_taylor:
        print("  gradient_checkpointing: OFF (act_taylor needs activation persistence)")

    skip_pats = [p for p in (args.skip_grad_patterns or "").split(",") if p]
    n_grad = 0
    n_skip = 0
    for name, p in model.named_parameters():
        if any(s in name for s in skip_pats):
            p.requires_grad_(False)
            n_skip += 1
        else:
            p.requires_grad_(True)
            n_grad += 1
    print(f"  grad params: {n_grad} tracked, {n_skip} skipped (patterns: {skip_pats})")

    extractor = HookedExtractor(model, device, with_act_taylor=args.with_act_taylor)
    print(f"  param tensors tracked: {len(extractor.acc_grad_l1)}")

    # ── Resume sidecar handling ────────────────────────────────────────────────
    config_hash = _compute_config_hash(
        model_path=str(args.model), samples_path=str(samples_file),
        max_samples=args.max_samples, max_len=args.max_len, chunk_len=args.chunk_len,
        skip_grad_patterns=args.skip_grad_patterns, with_act_taylor=args.with_act_taylor,
        task=args.task, prompt_key=preset["prompt_key"], completion_key=preset["completion_key"],
        pass_key=preset["pass_key"], pass_value=preset["pass_value"], keep_doc_ids=keep_ids,
    )
    sidecar = _sidecar_path(args.output)
    resume_skip = 0
    if args.resume:
        loaded = _load_sidecar_or_refuse(sidecar, config_hash, len(passing))
        if loaded == "REFUSE":
            sys.exit(2)
        if loaded is not None:
            extractor.load_state(loaded)
            resume_skip = int(loaded.get("next_sample_idx", 0))
            print(f"  RESUMED from {sidecar.name}: n_samples_done={extractor._n_samples} "
                  f"next_sample_idx={resume_skip}/{len(passing)}")
        else:
            print(f"  --resume set but no compatible sidecar at {sidecar} — starting fresh")
    elif sidecar.exists():
        stale = sidecar.with_suffix(sidecar.suffix + ".stale")
        sidecar.replace(stale)
        print(f"  no --resume; existing sidecar moved to {stale.name}")
    ckpt_dtype_t = {"float16": torch.float16, "float32": torch.float32}[args.ckpt_dtype]

    t0 = time.time()
    n_used = 0
    last_save_idx = resume_skip
    last_save_time = t0
    chunk_len = args.chunk_len if args.chunk_len > 0 else args.max_len
    if args.chunk_len > 0:
        print(f"  chunked grad accumulation: chunk_len={args.chunk_len}, hard cap max_len={args.max_len}")
    if resume_skip >= len(passing):
        print(f"  sidecar already covers all {len(passing)} samples — skipping loop, exporting only")
    iter_start = min(resume_skip, len(passing))
    for i, (prompt, comp) in enumerate(passing[iter_start:], start=iter_start):
        prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        comp_ids = tok(comp, return_tensors="pt", add_special_tokens=False).input_ids[0]
        full = torch.cat([prompt_ids, comp_ids], dim=0)
        if full.shape[0] > args.max_len:
            # keep tail of prompt + all comp if we can; otherwise truncate comp
            if comp_ids.shape[0] >= args.max_len:
                full = comp_ids[: args.max_len]
                start_comp = 0
            else:
                keep_prompt = args.max_len - comp_ids.shape[0]
                full = torch.cat([prompt_ids[-keep_prompt:], comp_ids], dim=0)
                start_comp = keep_prompt
        else:
            start_comp = prompt_ids.shape[0]
        if comp_ids.shape[0] < 2:
            continue

        seq_len = full.shape[0]
        # Build chunk windows over [0, seq_len). chunk_len defaults to max_len (=> 1 chunk = legacy path).
        chunks_used = 0
        n_chunks = (seq_len + chunk_len - 1) // chunk_len
        last_loss = float("nan")
        try:
            for ci in range(n_chunks):
                cs = ci * chunk_len
                ce = min(cs + chunk_len, seq_len)
                chunk_ids = full[cs:ce]
                chunk_labels = chunk_ids.clone()
                # mask any positions in this chunk that are still inside the prompt region
                mask_until = max(0, min(start_comp - cs, chunk_ids.shape[0]))
                if mask_until > 0:
                    chunk_labels[:mask_until] = -100
                if (chunk_labels != -100).sum().item() < 2:
                    continue  # nothing useful to backprop in this chunk

                input_ids = chunk_ids.unsqueeze(0).to(device)
                labels = chunk_labels.unsqueeze(0).to(device)
                out = model(input_ids=input_ids, labels=labels, use_cache=False)
                loss = out.loss
                if loss is None or not torch.isfinite(loss):
                    del out, input_ids, labels
                    continue
                loss.backward()
                last_loss = float(loss.item())
                chunks_used += 1
                del out, loss, input_ids, labels
                # Flush grads to CPU between chunks so the bf16 grad buffer (~= model size)
                # doesn't pin VRAM that the next chunk's activations need.
                if args.chunk_len > 0 and ci < n_chunks - 1:
                    extractor.flush_grads_to_cpu()

            if chunks_used > 0:
                extractor.flush_grads_to_cpu()
                extractor.mark_sample_done()
                n_used += 1
            else:
                # no usable chunk — make sure we don't carry stale grads
                for p in model.parameters():
                    p.grad = None

            # Sidecar checkpoint cadence (sample-based or wall-clock based, whichever fires first)
            samples_since_save = (i + 1) - last_save_idx
            elapsed_since_save = time.time() - last_save_time
            should_save = (
                (args.ckpt_every > 0 and samples_since_save >= args.ckpt_every) or
                (args.ckpt_time_sec > 0 and elapsed_since_save >= args.ckpt_time_sec)
            )
            if should_save and chunks_used > 0:
                _save_sidecar(sidecar, extractor, next_idx=i + 1,
                              passing_count=len(passing), config_hash=config_hash,
                              t0_wall=t0, ckpt_dtype=ckpt_dtype_t)
                last_save_idx = i + 1
                last_save_time = time.time()
                print(f"  ckpt: saved sidecar at sample {i+1}/{len(passing)} "
                      f"({sidecar.stat().st_size/1e9:.2f} GB)", flush=True)

            if (i+1) % 5 == 0 or i == 0:
                el = time.time() - t0
                tag = f"chunks={chunks_used}/{n_chunks}" if args.chunk_len > 0 else f"loss={last_loss:.4f}"
                print(f"  [{i+1}/{len(passing)}] seq={seq_len} {tag} elapsed={el:.1f}s "
                      f"vram={torch.cuda.memory_allocated()/1e9:.1f}G", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [{i+1}/{len(passing)}] OOM seq_len={seq_len} chunk_len={chunk_len} "
                  f"chunks_done={chunks_used}/{n_chunks} "
                  f"vram_alloc={torch.cuda.memory_allocated()/1e9:.1f}G "
                  f"vram_max={torch.cuda.max_memory_allocated()/1e9:.1f}G msg={str(e)[:200]}")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            for p in model.parameters():
                p.grad = None
            continue
        except Exception as e:
            print(f"  [{i+1}/{len(passing)}] error: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            for p in model.parameters():
                p.grad = None
            continue

    extractor.remove()
    print(f"  used {n_used}/{len(passing)} samples in {time.time()-t0:.1f}s")
    if n_used == 0:
        print("ERROR: no samples processed", file=sys.stderr)
        sys.exit(1)

    save_dtype_t = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.save_dtype]
    structured_cfg: Optional[Dict[str, int]] = None
    if args.structured:
        # Pull shape numbers from the loaded model config. text_config nested form
        # (e.g. Gemma 4) is checked first; fall back to top-level for plain models.
        cfg_obj = getattr(model, "config", None)
        text_cfg = getattr(cfg_obj, "text_config", None) or cfg_obj
        try:
            n_heads = int(getattr(text_cfg, "num_attention_heads"))
            n_kv = int(getattr(text_cfg, "num_key_value_heads", n_heads))
            hd = int(getattr(text_cfg, "head_dim",
                             getattr(text_cfg, "hidden_size") // n_heads))
            inter = int(getattr(text_cfg, "intermediate_size"))
            structured_cfg = {
                "num_heads": n_heads, "num_kv_heads": n_kv,
                "head_dim": hd, "intermediate_size": inter,
            }
            print(f"  structured_config: {structured_cfg}")
        except (AttributeError, TypeError) as e:
            print(f"  WARN: --structured requested but config probe failed ({type(e).__name__}: {e}); "
                  f"continuing without structured aggregates", file=sys.stderr)
            structured_cfg = None

    out_tensors = extractor.export(out_dtype=save_dtype_t, structured_config=structured_cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": str(args.model),
        "samples": str(samples_file),
        "task": args.task or "custom",
        "n_used": str(n_used),
        "n_total": str(len(passing)),
        "max_len": str(args.max_len),
        "chunk_len": str(args.chunk_len),
        "max_samples": str(args.max_samples),
        "extra": args.meta,
    }
    if structured_cfg is not None:
        metadata["structured_config"] = json.dumps(structured_cfg, sort_keys=True)
    save_file(out_tensors, str(args.output), metadata=metadata)
    print(f"  wrote {args.output} ({len(out_tensors)} tensors, "
          f"{args.output.stat().st_size/1e9:.2f} GB)")

    # Clean finish — discard the sidecar so a subsequent --resume on this output
    # won't pick up a stale partial accumulator.
    if sidecar.exists():
        try:
            sidecar.unlink()
            print(f"  cleaned sidecar {sidecar.name}")
        except OSError as e:
            print(f"  WARN: could not unlink sidecar {sidecar}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
