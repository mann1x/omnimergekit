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
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import save_file


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

    def export(self, out_dtype: torch.dtype = torch.float16) -> Dict[str, torch.Tensor]:
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

    t0 = time.time()
    n_used = 0
    chunk_len = args.chunk_len if args.chunk_len > 0 else args.max_len
    if args.chunk_len > 0:
        print(f"  chunked grad accumulation: chunk_len={args.chunk_len}, hard cap max_len={args.max_len}")
    for i, (prompt, comp) in enumerate(passing):
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
    out_tensors = extractor.export(out_dtype=save_dtype_t)
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
    save_file(out_tensors, str(args.output), metadata=metadata)
    print(f"  wrote {args.output} ({len(out_tensors)} tensors, "
          f"{args.output.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
