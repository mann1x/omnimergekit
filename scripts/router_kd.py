#!/usr/bin/env python3
"""router_kd.py — T18 Step 3: Router Knowledge Distillation (Router KD) for
recovering the gating network of a pruned fine-grained MoE.

Paper: Hyeon & Do, "Is Retraining-Free Enough? The Necessity of Router
       Calibration for Efficient MoE Compression" (arXiv:2603.02217).
       PDF archived at docs/papers/router_kd_2603.02217.pdf.

Recipe is taken VERBATIM from the paper's Appendix F.3 / Table 1 (read from
the downloaded PDF — the project memory's research-agent summary had the
temperature and learning rate WRONG; do not "fix" these back):

    calibration dataset   c4 (allenai/c4, en)
    epochs                1
    batch size            2
    grad accumulation     4              (effective batch 8)
    learning rate         5e-5
    max sequence length   512
    KD temperature  (tau)  1.0           (tau^2 == 1 -> plain masked fwd-KL)
    max calibration samples 3000         (~1.54M tokens)
    optimizer             AdamW, wd 0    (not stated in paper -> standard default)

Loss (paper Eq. 3), full vocabulary, forward KL teacher||student:

    L = (tau^2 / N_x) * sum_t  m_{t+1} * KL( p_T^t || p_S^t )

with p_M = softmax(z_M / tau) over R^|V|, m the next-token (padding) mask,
N_x = sum(m) + eps. Gradients flow EXCLUSIVELY to the student router
parameters; experts + backbone + embeddings + lm_head are frozen. Router KD
is drop-map-free (it distills output logits, not gate values), so no
--drop-map is needed — unlike router_eac_calibrate.py (Step 2).

Trainable theta_R for Gemma 4 = the three router tensors per MoE layer:
    router.proj.weight, router.scale, router.per_expert_scale
(--train-tensors {all,proj}; default all, ~0.04% of params).

Faithful, pod-only design (single A100-80GB, or 2x3090 split-by-model):
teacher 128e loaded bf16 frozen (~52 GB), student loaded 4-bit NF4 with the
router modules SKIPPED from quantization so they stay bf16-trainable (QLoRA
style; ~13 GB). Both default to device {"":0}. This cannot run on a single
24 GB 3090 — that's by design (the cached-top-K compromise was declined).

Reversibility (mirrors router_eac_calibrate.py): the calibrated router is
written into a NEW dir (--out-dir, default <variant>-rkd-it) so the source
bf16 is never clobbered; within that dir each touched shard is backed up to
<shard>.pre_router_kd and --restore reverts.

Usage:
    # train on a pod (A100-80GB), faithful recipe, default target = v5-coder:
    python scripts/router_kd.py \
        --base-dir   google/gemma-4-26B-A4B-it \
        --variant-dir google/gemma-4-A4B-98e-v5-coder-it \
        --out-dir    google/gemma-4-A4B-98e-v5-coder-rkd-it \
        --checkpoint-dir logs/router_kd_v5coder/ckpt \
        --canary-file scripts/ifeval_rumination_canaries.json --canary-gate

    # smoke (2 steps, 8 samples) — plumbing + freeze + write-back/restore check:
    python scripts/router_kd.py ... --max-samples 8 --max-steps 2 --no-canary

    # validate flags / dirs / VRAM without loading any weights:
    python scripts/router_kd.py ... --dry-run

    # revert a calibrated out-dir to its pre-KD shards:
    python scripts/router_kd.py --out-dir <dir> --restore
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

BACKUP_SUFFIX = ".pre_router_kd"
ROUTER_SUFFIXES = ("router.proj.weight", "router.scale", "router.per_expert_scale")
# T192 E-ExpertKD: the fused Gemma-4 expert tensors + the always-on dense mlp.
EXPERT_SUFFIXES = ("experts.gate_up_proj", "experts.down_proj")
SHARED_SUFFIXES = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


def _layer_of(name: str) -> int:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def _train_suffixes(train_tensors: str):
    """Map a --train-tensors choice to the tensor-name suffix tuple to unfreeze."""
    if train_tensors == "proj":
        return ("router.proj.weight",)
    if train_tensors in ("all", "router"):
        return ROUTER_SUFFIXES
    suff: tuple = ()
    if "experts" in train_tensors:
        suff += EXPERT_SUFFIXES
    if "router" in train_tensors:                 # experts+router
        suff += ROUTER_SUFFIXES
    if "shared" in train_tensors:                 # experts+shared
        suff += SHARED_SUFFIXES
    if not suff:
        raise SystemExit(f"FAIL: unknown --train-tensors {train_tensors!r}")
    return suff
# degenerate-loop detector, identical to the IFEval canary builder
LOOP_RE = re.compile(r"(.{2,40}?)\1{7,}|(.)\2{40,}", re.DOTALL)


# ─── shared helpers (no torch import — usable in --dry-run) ───────────────────

def find_router_tensors(model_dir: Path) -> Dict[str, List[Tuple[int, str]]]:
    """{shard_filename: [(layer_idx, tensor_name), ...]} for all 3 router tensors.

    Generalises router_eac_calibrate.find_router_proj_tensors to the full
    gating-parameter set (proj.weight + scale + per_expert_scale).
    """
    idx_path = model_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        raise SystemExit(f"FAIL: {idx_path} not found")
    with open(idx_path) as f:
        wm = json.load(f)["weight_map"]
    out: Dict[str, List[Tuple[int, str]]] = {}
    for name, shard in wm.items():
        if name.endswith(ROUTER_SUFFIXES):
            digits = [int(p) for p in name.split(".") if p.isdigit()]
            if not digits:
                continue
            out.setdefault(shard, []).append((digits[-1], name))
    return out


def _gpu_free_gib() -> float:
    """Best free single-GPU VRAM in GiB, via nvidia-smi (no torch needed)."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        return max(float(x) for x in out.split()) / 1024.0
    except Exception as ex:  # noqa: BLE001
        print(f"WARN: could not query nvidia-smi ({ex}); skipping VRAM check")
        return -1.0


# ─── --restore ────────────────────────────────────────────────────────────────

def do_restore(out_dir: Path) -> int:
    routers = find_router_tensors(out_dir)
    restored = 0
    for shard in routers:
        bak = out_dir / (shard + BACKUP_SUFFIX)
        if not bak.exists():
            print(f"WARN: {bak.name} missing — cannot restore {shard}")
            continue
        (out_dir / shard).write_bytes(bak.read_bytes())
        restored += 1
        print(f"  restored {shard}")
    print(f"OK restored {restored} shard(s) in {out_dir}")
    return 0 if restored else 1


# ─── --dry-run ──────────────────────────────────────────────────────────────

def do_dry_run(args: argparse.Namespace) -> int:
    base, var = Path(args.base_dir), Path(args.variant_dir)
    print("[dry-run] validating inputs (no weights loaded)")
    ok = True
    for label, d in (("base", base), ("variant", var)):
        idx = d / "model.safetensors.index.json"
        marker = "OK" if idx.exists() else "MISSING"
        if not idx.exists():
            ok = False
        print(f"  {label:8s} {d}  index:{marker}")
    if var.exists():
        rt = find_router_tensors(var)
        n = sum(len(v) for v in rt.values())
        layers = sorted({li for v in rt.values() for li, _ in v})
        print(f"  variant router tensors: {n} across {len(rt)} shard(s), "
              f"layers {layers[:3]}..{layers[-3:]} ({len(layers)} MoE layers)")
        if n != 3 * len(layers):
            print(f"  WARN: expected 3 router tensors/layer, got {n}/{len(layers)}")
    free = _gpu_free_gib()
    need = 70.0 if args.teacher_load == "bf16" else 35.0
    vram = "skip" if free < 0 else f"{free:.1f} GiB free (need ~{need:.0f})"
    if 0 <= free < need:
        ok = False
        print(f"  VRAM: {vram}  -> INSUFFICIENT for --teacher-load {args.teacher_load}")
    else:
        print(f"  VRAM: {vram}")
    print(f"  recipe: tau={args.tau} lr={args.lr} epochs={args.epochs} "
          f"bs={args.batch_size} ga={args.grad_accum} maxlen={args.max_seq_len} "
          f"samples={args.max_samples} train-tensors={args.train_tensors}")
    print(f"  out-dir: {args.out_dir}")
    print("[dry-run]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ─── model loading ────────────────────────────────────────────────────────────

_BNB_PATCHED = False


def _patch_bnb_for_accelerate():
    """accelerate 1.13.0 passes `_is_hf_initialized` to Params4bit.__new__ during
    dispatch_model; bitsandbytes 0.49.2 rejects it -> TypeError. Strip the kwarg.
    Same patch as scripts/pod_cache_nf4.py / convert_to_4bit.py; idempotent. Must
    run before any 4bit from_pretrained on this env (torch 2.11 / tf5 / acc 1.13
    / bnb 0.49.2)."""
    global _BNB_PATCHED
    if _BNB_PATCHED:
        return
    try:
        from bitsandbytes.nn.modules import Params4bit
    except Exception as e:  # bnb not importable -> bf16-only run, nothing to patch
        print(f"[load] bnb Params4bit patch skipped ({e})")
        return
    _orig_new = Params4bit.__new__

    def _patched_new(cls, *a, **k):
        k.pop("_is_hf_initialized", None)
        return _orig_new(cls, *a, **k)

    Params4bit.__new__ = _patched_new
    _BNB_PATCHED = True
    print("[load] patched bnb Params4bit.__new__ to drop _is_hf_initialized "
          "(accelerate 1.13 / bnb 0.49.2 compat)")


def _bnb_config(load: str):
    import torch
    from transformers import BitsAndBytesConfig
    if load == "bf16":
        return None
    return BitsAndBytesConfig(
        load_in_4bit=(load == "4bit"),
        load_in_8bit=(load == "8bit"),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        # keep the gating modules in bf16 so the router stays trainable.
        # lm_head + embed_tokens are TIED in Gemma-4: a live-4bit load can't form a
        # standalone Param4bit for the tied lm_head, so its FP4 quant-state is never
        # packed and the first teacher forward asserts (bnb modules.py:415
        # `weight.shape[1]==1`, "FP4 quantization state not initialized", bs2
        # 2026-06-02). Skipping the tied pair keeps it bf16 — fixes the crash AND
        # gives a higher-precision KD logit target (the teacher head we distill to).
        llm_int8_skip_modules=["router", "lm_head", "embed_tokens"],
        # let accelerate stage the transient bf16 materialization on CPU during
        # load. tf5 core_model_loading materializes each safetensor fragment in
        # FULL precision BEFORE bnb quantizes it; without this flag + device_map
        # =auto + max_memory the spike OOMs a 24GB card (router_kd hit 22.87GB at
        # 41% of weights, 2026-05-24). Proven in scripts/pod_cache_nf4.py on this
        # exact 2x3090 stack (council csl-2026-05-24-1637-067a).
        llm_int8_enable_fp32_cpu_offload=True,
    )


def _assert_router_real_dtype(model, load_mode: str) -> None:
    """Fail-fast guard (council csl-…-1d54 CONCERN #1).

    bnb only quantizes nn.Linear weights, so the only router tensor at risk is
    `router.proj.weight`; `router.scale` and `router.per_expert_scale` are bare
    nn.Parameters that bnb never touches and stay bf16 unconditionally. We still
    verify proj.weight survived as real float — but we do it right after the
    STUDENT load and BEFORE the ~52 GB teacher load, so a skip-modules miss
    aborts in student-load time (~minutes) instead of after both models are
    resident (the 'only fails at :248 after the 10-min load' complaint)."""
    import torch
    bad = [(n, str(p.dtype)) for n, p in model.named_parameters()
           if n.endswith(ROUTER_SUFFIXES)
           and p.dtype not in (torch.bfloat16, torch.float16, torch.float32)]
    if bad:
        ex = "; ".join(f"{n}:{d}" for n, d in bad[:3])
        raise SystemExit(
            f"FAIL (pre-teacher-load): {len(bad)} router tensor(s) quantized under "
            f"--student-load {load_mode} (e.g. {ex}). llm_int8_skip_modules=['router'] "
            f"did not protect router.proj.weight. Re-run with --student-load bf16 on a "
            f"2-GPU pod (--teacher-device '{{\"\":0}}' --student-device '{{\"\":1}}').")
    print(f"[load] router dtype OK (real float under --student-load {load_mode}) "
          f"— safe to load teacher")


def _dev_arg(s):
    """Parse a device override. A dict-style string like '{"":1}' is JSON-parsed
    into a real device_map dict — bitsandbytes 4-bit placement requires a dict
    (a bare 'cuda:1' string loads ~bf16 and OOMs). Plain strings pass through."""
    if s and s.strip().startswith("{"):
        import json
        return json.loads(s)
    return s


def _gpu_index_of(device_override):
    """Resolve a target GPU index from a {'':N} dict / 'cuda:N' string / None."""
    dev = _dev_arg(device_override) if device_override else None
    if isinstance(dev, dict) and dev:
        v = str(next(iter(dev.values())))
        return int(v) if v.isdigit() else (int(v.split(":")[-1]) if ":" in v else 0)
    if isinstance(dev, str) and dev.startswith("cuda") and ":" in dev:
        return int(dev.split(":")[-1])
    return 0


def _load_causal_lm(path, load_mode, device_override, args):
    """Load one model.

    bf16  -> plain dict device_map (no materialization spike on a big card).
    4bit/8bit -> device_map='auto' + per-GPU max_memory + offload_folder so the
    tf5 core_model_loading bf16 materialization transient spills to CPU instead
    of OOMing the target 24GB card. Pinning max_memory to ONE gpu keeps the
    student and teacher on separate cards (GPU1 / GPU0) while the ~11-14GB NF4
    steady-state stays fully resident there. Fix for the {'':N}-forces-all-on-
    one-card OOM (router_kd 22.87GB@41%, 2026-05-24; council csl-…-1637-067a;
    proven pattern in scripts/pod_cache_nf4.py on this 2x3090 stack)."""
    import json
    import torch
    from transformers import AutoModelForCausalLM
    common = dict(dtype=torch.bfloat16, trust_remote_code=True,
                  low_cpu_mem_usage=True, attn_implementation=args.attn_impl)

    # Pre-quantized NF4 dir (config.json already carries quantization_config):
    # on-disk weights are 4bit, so there's NO bf16 materialization spike and
    # accelerate sizes by the real ~11-14GB (not the bf16 footprint). Load straight
    # to the target GPU with a dict device_map — fits resident, no CPU/disk offload,
    # no torch-2.11 meta crash. This is the route that actually works on 2x3090;
    # a LIVE 4bit load is structurally blocked by accelerate's bf16-size planning
    # (teacher 49GB bf16 > 44GB aggregate -> forced offload -> meta).
    cfgp = Path(path) / "config.json"
    is_prequant = False
    if cfgp.exists():
        try:
            is_prequant = "quantization_config" in json.loads(cfgp.read_text())
        except Exception:
            is_prequant = False
    if is_prequant:
        _patch_bnb_for_accelerate()
        gi = _gpu_index_of(device_override)
        print(f"[load]   pre-quantized NF4 -> device_map={{'': {gi}}} (fits resident, no spike)")
        return AutoModelForCausalLM.from_pretrained(path, device_map={"": gi}, **common)

    if load_mode == "bf16":
        dev = {"": 0} if args.device_map == "single" else args.device_map
        return AutoModelForCausalLM.from_pretrained(
            path,
            device_map=_dev_arg(device_override) if device_override else dev,
            **common)
    _patch_bnb_for_accelerate()
    gi = _gpu_index_of(device_override)
    # CPU budget gives the tf5 bf16 materialization transient somewhere to stage
    # (fixes the all-on-one-card 22.87GB OOM). NO offload_folder on purpose: on
    # torch 2.11 a DISK-offloaded bnb-4bit module crashes accelerate's
    # execution-device-hook -> state_dict -> _save_to_state_dict ->
    # quant_state.offset.item() with "cannot be called on meta tensors" (the
    # double-quant nested offset is left on meta). Withholding offload_folder
    # forbids disk offload; the ~11-14GB NF4 fits the 22GB GPU budget so nothing
    # offloads in steady state (council csl-…-1637-067a + pod_cache_nf4 recipe,
    # adapted for the torch-2.11 meta regression).
    max_memory = {gi: f"{args.gpu_mem_gib}GiB", "cpu": f"{args.cpu_mem_gib}GiB"}
    print(f"[load]   {load_mode} via device_map=auto gpu={gi} max_memory={max_memory}")
    return AutoModelForCausalLM.from_pretrained(
        path,
        quantization_config=_bnb_config(load_mode),
        device_map="auto",
        max_memory=max_memory,
        **common)


def load_models(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Student (the quantized one) loads FIRST so its router dtype is verified
    # before we commit the big teacher load (fail-fast, council CONCERN #1).
    print(f"[load] student variant ({args.student_load}) from {args.variant_dir}")
    t0 = time.time()
    student = _load_causal_lm(args.variant_dir, args.student_load,
                              args.student_device, args)
    print(f"[load] student in {time.time()-t0:.0f}s")
    _assert_router_real_dtype(student, args.student_load)

    print(f"[load] teacher 128e ({args.teacher_load}) frozen from {args.base_dir}")
    t0 = time.time()
    teacher = _load_causal_lm(args.base_dir, args.teacher_load,
                              args.teacher_device, args)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"[load] teacher in {time.time()-t0:.0f}s")
    return tok, teacher, student


def select_router_params(student, train_tensors: str, train_layers: str = "all"):
    """Freeze everything; unfreeze the selected tensors (router and/or experts
    and/or shared mlp), optionally restricted to a layer band. Asserts the
    unfrozen params are real float (not 4/8-bit) so they can actually train."""
    import torch
    want = _train_suffixes(train_tensors)
    if train_layers in ("all", "", None):
        lset = None
    elif train_layers == "mid":
        lset = set(range(8, 23))      # L8-22 — rank-probe's most-absorptive band
    else:
        lset = {int(x) for x in str(train_layers).split(",")}

    trainable, names = [], []
    for n, p in student.named_parameters():
        hit = n.endswith(want) and (lset is None or _layer_of(n) in lset)
        if hit:
            p.requires_grad_(True)
            trainable.append(p)
            names.append(n)
            # trainable tensors MUST be real float (not 4/8-bit) to optimize
            if p.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise SystemExit(
                    f"FAIL: trainable param {n} has dtype {p.dtype} — it was "
                    f"quantized. Load the student bf16 (--student-load bf16); "
                    f"experts/router must be real float to train.")
        else:
            p.requires_grad_(False)

    total = sum(p.numel() for p in student.parameters())
    train_n = sum(p.numel() for p in trainable)
    print(f"[freeze] train_tensors={train_tensors} layers={train_layers}: "
          f"{len(names)} tensors, {train_n:,} elems = {100*train_n/total:.4f}% "
          f"of {total:,}")
    if not trainable:
        raise SystemExit("FAIL: no params selected — check --train-tensors/--train-layers")
    return trainable, names


# ─── data ─────────────────────────────────────────────────────────────────────

def _iter_texts(args):
    """Yield calibration document strings.

    Default: stream allenai/c4 (the paper recipe). If --corpus-file is given
    (a JSONL with a `text` field per line, e.g. scripts/router_calib_corpus.jsonl
    from build_router_calib_corpus.py), iterate those domain-matched docs first;
    with --corpus-pad-c4 the remaining budget up to --max-samples is topped up
    from C4 so KD still sees the paper's ~1.5M-token regime."""
    n = 0
    if args.corpus_file:
        cpath = Path(args.corpus_file)
        if not cpath.exists():
            raise SystemExit(f"FAIL: --corpus-file {cpath} not found")
        print(f"[data] domain corpus {cpath}")
        for line in cpath.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line).get("text", "")
            except Exception:  # noqa: BLE001
                t = line
            if t.strip():
                yield t
                n += 1
                if n >= args.max_samples:
                    return
        print(f"[data] domain corpus yielded {n} docs")
        if not args.corpus_pad_c4:
            return
        print(f"[data] padding to {args.max_samples} with C4 (--corpus-pad-c4)")
    from datasets import load_dataset
    print(f"[data] streaming allenai/c4 en, taking {args.max_samples - n} docs")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    for ex in ds:
        text = ex.get("text", "")
        if not text.strip():
            continue
        yield text
        n += 1
        if n >= args.max_samples:
            break


def build_batches(tok, args):
    """Yield (input_ids, attention_mask) tensors of shape (B, L)."""
    import torch
    buf_ids, buf_mask = [], []
    for text in _iter_texts(args):
        enc = tok(text, truncation=True, max_length=args.max_seq_len,
                  padding="max_length", return_tensors="pt")
        buf_ids.append(enc["input_ids"][0])
        buf_mask.append(enc["attention_mask"][0])
        if len(buf_ids) == args.batch_size:
            yield torch.stack(buf_ids), torch.stack(buf_mask)
            buf_ids, buf_mask = [], []
    if buf_ids:
        yield torch.stack(buf_ids), torch.stack(buf_mask)


# ─── KD loss ──────────────────────────────────────────────────────────────────

def kd_loss(t_logits, s_logits, attn_mask, tau: float, eps: float = 1e-6):
    """Masked full-vocab forward-KL, paper Eq. 3 (tau^2-scaled)."""
    import torch
    import torch.nn.functional as F
    # next-token: predict t+1 from <=t
    t = t_logits[:, :-1, :].float()
    s = s_logits[:, :-1, :].float()
    m = attn_mask[:, 1:].float()                      # (B, L-1) target mask
    flat = m.reshape(-1) > 0
    if flat.sum() == 0:
        return s.sum() * 0.0
    t = t.reshape(-1, t.shape[-1])[flat]              # (P, V)
    s = s.reshape(-1, s.shape[-1])[flat]
    with torch.no_grad():
        p_t = F.softmax(t / tau, dim=-1)
    logp_s = F.log_softmax(s / tau, dim=-1)
    # KL(p_T || p_S) = sum p_T (log p_T - log p_S); reduction batchmean over P
    kl = F.kl_div(logp_s, p_t, reduction="none").sum(-1)   # (P,)
    n_x = kl.numel() + eps
    return (tau * tau) * kl.sum() / n_x


# ─── generation canary (relative AR-mode signal before write-back) ────────────

def run_canary(tok, model, canary_path: Path, max_new: int, tag: str,
               dump_db: str | None = None):
    import torch
    docs = json.load(open(canary_path))
    db = None
    if dump_db:
        import datetime as _dt
        import sqlite3
        db = sqlite3.connect(dump_db)
        db.execute("CREATE TABLE IF NOT EXISTS answers (doc_id TEXT, tag TEXT, "
                   "chars INT, loop INT, ref_128e INT, prompt TEXT, response TEXT, "
                   "ts TEXT)")
        db.commit()
        print(f"  [canary:{tag}] dumping answers to {dump_db} (live-peekable)")
    rows = []
    for d in docs:
        msgs = [{"role": "user", "content": d["prompt"]}]
        # transformers 5.x apply_chat_template returns a BatchEncoding dict, not a
        # bare tensor — request return_dict and splat into generate(**enc), else
        # generate() does inputs_tensor.shape[0] on the dict and AttributeErrors.
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        # inputs go on the input-embedding's device (robust whether the model is
        # single-device or device_map-dispatched across GPUs).
        in_dev = model.get_input_embeddings().weight.device
        enc = {k: v.to(in_dev) for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                  temperature=None, top_p=None, top_k=None,
                                  pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][in_len:], skip_special_tokens=True)
        looped = bool(LOOP_RE.search(text))
        rows.append({"doc_id": d["doc_id"], "chars": len(text), "loop": looped,
                     "ref_128e": d.get("ref_128e_chars")})
        print(f"  [canary:{tag}] doc {d['doc_id']:>3}: {len(text):>6} chars "
              f"loop={looped} (128e ref {d.get('ref_128e_chars')})")
        if db is not None:
            db.execute(
                "INSERT INTO answers VALUES (?,?,?,?,?,?,?,?)",
                (str(d["doc_id"]), tag, len(text), int(looped),
                 d.get("ref_128e_chars"), d["prompt"], text,
                 _dt.datetime.now().isoformat()))
            db.commit()
    return rows


def canary_verdict(pre, post) -> bool:
    """True == OK to save: rumination did not worsen on any canary prompt."""
    ok = True
    for a, b in zip(pre, post):
        if b["loop"] and not a["loop"]:
            print(f"  GATE FAIL doc {b['doc_id']}: introduced a loop")
            ok = False
        if b["chars"] > max(a["chars"] * 1.10, a["chars"] + 200):
            print(f"  GATE FAIL doc {b['doc_id']}: {a['chars']}->{b['chars']} chars "
                  f"(>10% longer)")
            ok = False
    return ok


# ─── write-back ───────────────────────────────────────────────────────────────

def write_back(student, names, base_variant_dir: Path, out_dir: Path):
    """Copy variant dir -> out_dir (once), then overwrite the 3 router tensors
    per touched shard with the trained values, backing each shard up first."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    if not out_dir.exists():
        print(f"[save] copying {base_variant_dir} -> {out_dir}")
        shutil.copytree(base_variant_dir, out_dir)

    # name -> trained tensor (cpu)
    trained = {n: p.detach().to("cpu") for n, p in student.named_parameters()
               if n in set(names)}
    routers = find_router_tensors(out_dir)   # {shard: [(li, name)]}

    for shard, entries in routers.items():
        names_here = [nm for _, nm in entries if nm in trained]
        if not names_here:
            continue
        shard_path = out_dir / shard
        bak = out_dir / (shard + BACKUP_SUFFIX)
        if not bak.exists():
            print(f"[save] backup {shard} -> {bak.name}")
            bak.write_bytes(shard_path.read_bytes())
        meta, tensors = {}, {}
        with safe_open(str(shard_path), framework="pt") as f:
            meta = dict(f.metadata() or {})
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        for nm in names_here:
            tensors[nm] = trained[nm].to(tensors[nm].dtype)
        save_file(tensors, str(shard_path), metadata=meta or None)
        print(f"[save] rewrote {len(names_here)} router tensor(s) in {shard}")
    print(f"[save] DONE -> {out_dir}")


def write_back_general(student, names, base_variant_dir: Path, out_dir: Path):
    """Generalised write-back for ANY trained tensor set (experts/router/shared).
    Copies the student dir once, then overwrites each trained tensor in its own
    shard, resolved via the safetensors index. No per-shard backup: out_dir is a
    fresh disposable copy (the source variant_dir is never touched), so the T192
    sweep can `rm -rf out_dir` between runs without losing the A2 source."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    if not out_dir.exists():
        print(f"[save] copying {base_variant_dir} -> {out_dir}")
        shutil.copytree(base_variant_dir, out_dir)
    nameset = set(names)
    trained = {n: p.detach().to("cpu") for n, p in student.named_parameters()
               if n in nameset}
    idx_path = out_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        raise SystemExit(f"FAIL: {idx_path} not found")
    wm = json.load(open(idx_path))["weight_map"]
    by_shard: Dict[str, List[str]] = {}
    for n in trained:
        if n not in wm:
            raise SystemExit(f"FAIL: trained tensor {n} absent from weight_map")
        by_shard.setdefault(wm[n], []).append(n)
    for shard, ns in by_shard.items():
        sp = out_dir / shard
        with safe_open(str(sp), framework="pt") as f:
            meta = dict(f.metadata() or {})
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        for n in ns:
            tensors[n] = trained[n].to(tensors[n].dtype)
        save_file(tensors, str(sp), metadata=meta or None)
        print(f"[save] rewrote {len(ns)} tensor(s) in {shard}")
    print(f"[save] DONE ({len(trained)} tensors across {len(by_shard)} shards) -> {out_dir}")


# ─── training ─────────────────────────────────────────────────────────────────

def train(args) -> int:
    import torch

    tok, teacher, student = load_models(args)
    trainable, names = select_router_params(student, args.train_tensors, args.train_layers)
    if args.grad_checkpointing:
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        student.config.use_cache = False
        print("[train] gradient checkpointing enabled (use_reentrant=False)")
    if args.optim in ("adamw8bit", "paged_adamw8bit"):
        import bitsandbytes as bnb
        # PagedAdamW8bit pages the optimizer states (m, v) to CPU unified memory,
        # so the ~11GB first-opt.step() GPU spike for a multi-billion-param trainable
        # set never lands on-device. Needed for single-card E-ExpertKD (full survivor
        # experts): plain AdamW8bit OOMs at step 2 by ~1GB on one 97GB card with
        # teacher+student+grads+full-vocab-KL transients already resident (bs2
        # 2026-06-02). Harmless on the 2-card split.
        cls = bnb.optim.PagedAdamW8bit if args.optim == "paged_adamw8bit" else bnb.optim.AdamW8bit
        opt = cls(trainable, lr=args.lr, weight_decay=args.weight_decay)
        print(f"[train] optimizer: bitsandbytes {cls.__name__} lr={args.lr}")
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save_ckpt(step: int):
        if not ckpt_dir:
            return
        sd = {n: p.detach().cpu() for n, p in student.named_parameters() if n in set(names)}
        path = ckpt_dir / f"router_step{step:06d}.pt"
        torch.save({"step": step, "names": names, "router": sd,
                    "recipe": vars(args)}, path)
        print(f"[ckpt] saved {path.name} ({len(sd)} tensors)")

    # pre-train canary (baseline AR signal) — before any optimisation
    pre = None
    if args.canary_file and not args.no_canary:
        print("[canary] pre-train baseline:")
        pre = run_canary(tok, student, Path(args.canary_file), args.canary_max_new, "pre")

    student.train()
    micro = 0
    step = 0
    running = 0.0
    t0 = time.time()
    for epoch in range(args.epochs):
        for input_ids, attn in build_batches(tok, args):
            input_ids = input_ids.to(teacher.device)
            attn = attn.to(teacher.device)
            # Gemma 4 (transformers ≥5.5.0) gates create_causal_mask_mapping on
            # mm_token_type_ids whenever `self.training=True` (see
            # modeling_gemma4.py:2005 — unconditional `ValueError` if absent).
            # `.generate()` auto-fills this at generation/utils.py:924-927 but
            # `.forward()` does not. Student is in `.train()` from line 612
            # onward; without this kwarg the very first KD step crashes BEFORE
            # any loss is computed. Text-only sequences = all zeros (image
            # token type would be 1). Teacher is in `.eval()` (line 357) so
            # the check skips, but we pass zeros to both for symmetry.
            mm_zeros = torch.zeros_like(input_ids)
            with torch.no_grad():
                t_logits = teacher(input_ids=input_ids, attention_mask=attn,
                                   mm_token_type_ids=mm_zeros,
                                   use_cache=False).logits
            s_logits = student(input_ids=input_ids.to(student.device),
                               attention_mask=attn.to(student.device),
                               mm_token_type_ids=mm_zeros.to(student.device),
                               use_cache=False).logits
            loss = kd_loss(t_logits.to(s_logits.device), s_logits, attn.to(s_logits.device),
                           args.tau) / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum
            micro += 1
            if micro % args.grad_accum == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                avg = running / args.grad_accum
                running = 0.0
                if step % args.log_every == 0 or step == 1:
                    print(f"  step {step:5d} loss={avg:.5f} "
                          f"({(time.time()-t0)/step:.1f}s/step)", flush=True)
                if args.save_every and step % args.save_every == 0:
                    save_ckpt(step)
                if args.max_steps and step >= args.max_steps:
                    print(f"[train] hit --max-steps {args.max_steps}")
                    break
        if args.max_steps and step >= args.max_steps:
            break

    print(f"[train] done: {step} optimizer steps in {time.time()-t0:.0f}s")
    save_ckpt(step)

    # post-train canary + gate
    if pre is not None:
        print("[canary] post-train:")
        student.eval()
        post = run_canary(tok, student, Path(args.canary_file), args.canary_max_new, "post")
        ok = canary_verdict(pre, post)
        print(f"[canary] verdict: {'PASS' if ok else 'FAIL (rumination worse)'}")
        if args.canary_gate and not ok:
            print("[save] SKIPPED write-back — canary gate failed "
                  "(router checkpoint preserved; rerun with --no-canary to force)")
            return 2

    if args.no_save:
        print("[save] --no-save set; skipping write-back (checkpoint kept)")
        return 0
    if args.train_tensors in ("all", "proj", "router"):
        write_back(student, names, Path(args.variant_dir), Path(args.out_dir))
    else:
        write_back_general(student, names, Path(args.variant_dir), Path(args.out_dir))
    return 0


# ─── entry ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", help="Teacher: base 128e model dir (HF, bf16)")
    ap.add_argument("--variant-dir", help="Student: pruned 98e variant dir (HF, bf16)")
    ap.add_argument("--out-dir", help="Output dir for the calibrated student "
                    "(default <variant>-rkd-it). Source is never clobbered.")
    # recipe (paper Table 1 — defaults ARE the paper values; change with care)
    ap.add_argument("--tau", type=float, default=1.0, help="KD temperature (paper 1.0)")
    ap.add_argument("--lr", type=float, default=5e-5, help="AdamW lr (paper 5e-5)")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=3000)
    ap.add_argument("--corpus-file", default=None,
                    help="JSONL with a 'text' field per line (domain-matched "
                         "calib corpus from build_router_calib_corpus.py). "
                         "Default None = paper-faithful C4 stream.")
    ap.add_argument("--corpus-pad-c4", action="store_true",
                    help="after the domain corpus is exhausted, top up to "
                         "--max-samples with C4 (keeps the paper token regime)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--train-tensors",
                    choices=["all", "proj", "router", "experts",
                             "experts+router", "experts+shared"],
                    default="all",
                    help="all/router = router 3 tensors (paper); proj = proj.weight; "
                         "experts = the 62 survivor expert tensors (E-ExpertKD); "
                         "experts+router / experts+shared add those")
    ap.add_argument("--train-layers", default="all",
                    help="all | mid (L8-22) | comma list e.g. 8,12,18 — restrict "
                         "which layers' selected tensors train")
    ap.add_argument("--optim", choices=["adamw", "adamw8bit", "paged_adamw8bit"], default="adamw",
                    help="adamw8bit (bitsandbytes) for full-expert FT to fit VRAM; "
                         "paged_adamw8bit additionally pages optimizer states to CPU "
                         "(single-card full-expert KD — avoids the step-2 OOM spike)")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="enable gradient checkpointing (needed for full-expert FT)")
    # loading / hardware
    ap.add_argument("--teacher-load", choices=["bf16", "8bit", "4bit"], default="bf16")
    ap.add_argument("--student-load", choices=["bf16", "8bit", "4bit"], default="4bit")
    ap.add_argument("--device-map", default="single",
                    help="'single' -> {'':0}; or an HF device_map string/'auto'")
    ap.add_argument("--teacher-device", default=None,
                    help="Override teacher device_map (e.g. {'':0} for 2-GPU split)")
    ap.add_argument("--student-device", default=None,
                    help="Override student device_map (e.g. {'':1} for 2-GPU split)")
    ap.add_argument("--attn-impl", default="eager",
                    help="attn_implementation (eager is safest for Gemma 4 logits)")
    ap.add_argument("--gpu-mem-gib", type=int, default=22,
                    help="per-GPU max_memory budget for 4bit/8bit loads; the NF4 "
                         "model (~11-14GB) sits resident, CPU stages the spike")
    ap.add_argument("--cpu-mem-gib", type=int, default=200,
                    help="CPU max_memory headroom for the load-time bf16 transient")
    ap.add_argument("--offload-folder", default="/workspace/offload",
                    help="disk offload dir accelerate may use during the NF4 load")
    # logging / checkpoint
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--save-every", type=int, default=100, help="optimizer steps")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epoch")
    # canary
    ap.add_argument("--canary-file", default="scripts/ifeval_rumination_canaries.json")
    ap.add_argument("--canary-max-new", type=int, default=4096)
    ap.add_argument("--canary-dump", default=None,
                    help="sqlite db path; write each canary answer (doc_id, tag, "
                         "prompt, full response, chars, loop) as it completes, so "
                         "pre/post answers can be peeked live during training")
    ap.add_argument("--canary-gate", action="store_true",
                    help="refuse write-back if rumination worsens on any canary")
    ap.add_argument("--no-canary", action="store_true")
    # modes
    ap.add_argument("--dry-run", action="store_true",
                    help="validate dirs/flags/VRAM, load nothing")
    ap.add_argument("--restore", action="store_true",
                    help="restore --out-dir shards from .pre_router_kd backups")
    ap.add_argument("--no-save", action="store_true",
                    help="train + canary but skip write-back (checkpoint only)")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.out_dir is None and args.variant_dir:
        v = args.variant_dir.rstrip("/")
        args.out_dir = (v[:-3] + "-rkd-it") if v.endswith("-it") else (v + "-rkd")

    if args.restore:
        if not args.out_dir:
            print("FAIL: --restore needs --out-dir")
            return 1
        return do_restore(Path(args.out_dir))

    if not args.base_dir or not args.variant_dir:
        print("FAIL: --base-dir and --variant-dir are required")
        return 1

    if args.dry_run:
        return do_dry_run(args)

    return train(args)


if __name__ == "__main__":
    sys.exit(main())
