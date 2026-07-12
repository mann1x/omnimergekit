#!/usr/bin/env python3
"""router_eac_calibrate.py — T18 Step 2: EAC-MoE / QESC layer-by-layer
TopK-MSE router calibration on a pruned MoE variant.

Paper: "EAC-MoE: Expert-Selection Aware Compressor for MoE LLMs"
       (arXiv:2508.01625, ACL 2025). PDF: docs/papers/eac_moe_2508.01625.pdf.

AUDITED + CORRECTED 2026-05-24 against the paper AND Gemma 4's real router
forward (transformers `Gemma4TextRouter`). Three fixes vs the original
agent-built version (which was NEVER run on a variant, so nothing shipped):

  FIX #1 — capture point + router transform (was a correctness bug).
    Gemma 4's router consumes `residual` (the PRE-feedforward hidden state,
    decoder line ~1381) and applies its OWN `RMSNorm(with_scale=False)` then
    `* router.scale * hidden^-0.5` BEFORE `router.proj`. The original hooked
    `pre_feedforward_layernorm`'s OUTPUT and used it raw as `h @ proj.W.T` —
    a tensor the router never consumes, omitting the router's internal
    norm+scale. We now capture the TRUE proj input via a `forward_pre_hook`
    on each `router.proj` (that input already includes norm+scale, which is
    identical between teacher and variant), so `proj_input @ W.T` reproduces
    the real router logits exactly.

  FIX #2 — calibration window K (was a paper-default mismatch).
    The paper's TopK-MSE (Eq. 5, Fig. 4) needs K WIDER than native top-k to
    cover shifted experts (Deepseek 6/64 → top-16 covers 95.9% of shifts).
    Default is now `--calib-k 16` (was 8 == Gemma's native top-k). This is the
    calibration window, NOT the model's native top-k.

  FIX #3 — paired forward, x vs x̂ (was the acknowledged simplification).
    EAC matches `W x̂` (perturbed input) to `W_fp x` (full-precision logits).
    The original used base-128e h for BOTH teacher and student, never modelling
    the variant's drifted router input — the very drift drop-recovery must fix.
    We now forward BOTH models on the SAME calibration tokens and pair
    per-token: the teacher's top-K + target logit VALUES come from base-h via
    the base proj weights; the student MSE input is the VARIANT's own router
    input (h_var) via the variant proj weights being optimized.

EAC TopK-MSE loss (Eq. 5):
    L = (1/K) Σ_{i ∈ top-K(W x)}  ((W x̂)_i − (W x)_i)²
matched on logit VALUES at the teacher's top-K positions. Selection (top-k) is
identical on logits or softmax (monotonic), so logit-top-K is the right mask.
`per_expert_scale` is applied AFTER top-k (to the mixing weights), so it does
not affect selection — we therefore calibrate only `router.proj.weight`, and
leave `router.scale` / `router.per_expert_scale` untouched.

Two phases (run separately so one capture serves many calibrate runs):

  --phase capture   (needs --base-dir AND --variant-dir)
      Tokenize WikiText-2 ONCE, then forward base 128e and the variant on the
      SAME tokens (device_map=auto offload OK on a 3090). A forward_pre_hook on
      each `router.proj` streams the proj inputs to
      <cache>/h_base_layer_NN_batch_MM.pt and <cache>/h_var_layer_NN_batch_MM.pt.
      Per-token rows are aligned across the two caches.

  --phase calibrate (needs --base-dir, --variant-dir, --drop-map)
      Per layer: teacher top-K + targets from h_base @ base_W.T (128-space),
      map dropped positions out, optimize the VARIANT proj.weight so
      h_var @ var_W.T matches those targets at surviving positions (AdamW
      TopK-MSE). Write back to variant safetensors.

  --phase both       capture, then calibrate.

Backup: each variant shard touched is copied to <shard>.pre_eac_calibrate
before edit. --restore reverts.

Usage:
    # Capture once (base + variant on the same tokens; ~45 min on a 3090):
    python scripts/router_eac_calibrate.py --phase capture \\
        --base-dir google/gemma-4-26B-A4B-it \\
        --variant-dir google/gemma-4-A4B-98e-v5-coder-it \\
        --cache-dir /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/eac_cache \\
        --n-seq 128 --seq-len 2048

    # Then calibrate (~10-15 min on a 3090):
    python scripts/router_eac_calibrate.py --phase calibrate \\
        --base-dir google/gemma-4-26B-A4B-it \\
        --variant-dir google/gemma-4-A4B-98e-v5-coder-it \\
        --drop-map  scripts/v5coder_C6_v4floor_perlayer_breadth50_drop_map.json \\
        --cache-dir /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/eac_cache \\
        --calib-k 16 --lr 1e-3 --steps 100 --opt-batch-size 16384

NB: cache-dir is reusable across calibrate runs (same base+variant+data → same h).
NEVER point --cache-dir at /tmp (tmpfs) — the caches are tens of GB.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BACKUP_SUFFIX = ".pre_eac_calibrate"


# ─── shared helpers ──────────────────────────────────────────────────────────

def find_router_proj_tensors(model_dir: Path) -> Dict[int, Tuple[str, str]]:
    """Returns {layer_idx: (tensor_name, shard_filename)} for router.proj.weight."""
    idx_path = model_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        raise SystemExit(f"FAIL: {idx_path} not found")
    with open(idx_path) as f:
        wm = json.load(f)["weight_map"]
    out = {}
    for k, shard in wm.items():
        if k.endswith("router.proj.weight"):
            digits = [int(p) for p in k.split(".") if p.isdigit()]
            if not digits:
                continue
            out[digits[-1]] = (k, shard)
    return out


def load_base_router_weights(base_dir: Path) -> Dict[int, torch.Tensor]:
    """Returns {layer_idx: (128, hidden_size) FP32 tensor}."""
    tensors = find_router_proj_tensors(base_dir)
    out = {}
    for li, (name, shard) in tensors.items():
        with safe_open(str(base_dir / shard), framework="pt") as f:
            out[li] = f.get_tensor(name).to(torch.float32)
    return out


def _resolve_device_map(mdir: Path, max_memory, dtype):
    """Pick a device_map for from_pretrained.

    With no max_memory cap, plain "auto" is fine. WITH a cap, transformers'
    stock auto-infer ORPHANS a few top-level Gemma 4 buffers
    (model.vision_tower.std_bias / std_scale) and `check_device_map` then
    raises 'does not give any device for ...' (pod 37588132, 2026-05-24).
    We rebuild the map on a meta copy via infer_auto_device_map and assign any
    param/buffer the map doesn't cover to GPU 0 (those buffers are tiny)."""
    if not max_memory:
        return "auto"
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        from accelerate import infer_auto_device_map, init_empty_weights
    except Exception as ex:  # noqa: BLE001
        print(f"[capture] accelerate unavailable ({ex}); falling back to 'auto'")
        return "auto"
    cfg = AutoConfig.from_pretrained(str(mdir), trust_remote_code=True)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    nosplit = list(getattr(m, "_no_split_modules", None) or [])
    dmap = infer_auto_device_map(m, max_memory=max_memory,
                                 no_split_module_classes=nosplit, dtype=dtype)
    covered = list(dmap.keys())

    def _covered(name: str) -> bool:
        return any(name == k or name.startswith(k + ".") for k in covered)

    orphans = [n for n, _ in (list(m.named_parameters()) + list(m.named_buffers()))
               if not _covered(n)]
    for n in orphans:
        dmap[n] = 0
    if orphans:
        print(f"[capture] patched {len(orphans)} orphaned param/buffer -> gpu0: "
              f"{orphans[:4]}")
    del m
    return dmap


# ─── Phase 1: capture router-proj INPUTS via forward_pre_hook ────────────────

def _tokenize_calib(tok, args) -> "torch.Tensor":
    """Tokenize the calibration corpus ONCE → (n_seq, seq_len) int tensor.
    The same tensor is fed to base and variant so per-token rows align."""
    if args.corpus_file:
        cpath = Path(args.corpus_file)
        if not cpath.exists():
            raise SystemExit(f"FAIL: --corpus-file {cpath} not found")
        print(f"[capture] task corpus {cpath} ({cpath.stat().st_size/1e6:.1f} MB)")
        text = cpath.read_text()
    else:
        from datasets import load_dataset
        print("[capture] WikiText-2 raw (train split)...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(s for s in ds["text"] if s.strip())
    print(f"[capture] tokenizing {len(text):,} chars...")
    enc = tok(text, return_tensors="pt", truncation=False)["input_ids"][0]
    needed = args.n_seq * args.seq_len
    if enc.numel() < needed:
        print(f"WARN: only {enc.numel()} tokens, need {needed}; reducing n_seq")
        args.n_seq = enc.numel() // args.seq_len
    enc = enc[:args.n_seq * args.seq_len].view(args.n_seq, args.seq_len)
    print(f"[capture] {args.n_seq} seq × {args.seq_len} tok = {enc.numel():,} tokens")
    return enc


def _capture_one(model, enc, cache_dir: Path, prefix: str, batch_size: int) -> Dict[int, int]:
    """forward_pre_hook on every `router.proj`; stream the proj INPUT per batch.

    The proj input is `router.norm(residual) * router.scale * hidden^-0.5` —
    i.e. the exact pre-image of the real router logits (FIX #1). Returns the
    per-layer batch counter."""
    counters: Dict[int, int] = {}
    handles = []

    def make_hook(li: int):
        def pre_hook(_module, inputs):
            x = inputs[0]
            bi = counters[li]
            t = x.detach().to("cpu", torch.bfloat16).reshape(-1, x.shape[-1]).clone()
            torch.save(t, cache_dir / f"{prefix}_layer_{li:02d}_batch_{bi:04d}.pt")
            counters[li] = bi + 1
        return pre_hook

    n = 0
    for name, mod in model.named_modules():
        if name.endswith(".router.proj"):
            digits = [int(p) for p in name.split(".") if p.isdigit()]
            if not digits:
                continue
            li = digits[-1]
            handles.append(mod.register_forward_pre_hook(make_hook(li)))
            counters[li] = 0
            n += 1
    print(f"[capture:{prefix}] hooks on {n} router.proj modules: "
          f"{sorted(counters)[:3]}..{sorted(counters)[-3:]}")
    if n == 0:
        raise SystemExit(f"FAIL: no '.router.proj' modules in {prefix} model — "
                         f"check architecture (expected Gemma4TextRouter)")

    device = next(model.parameters()).device
    for i in range(0, enc.shape[0], batch_size):
        t0 = time.time()
        batch = enc[i:i + batch_size].to(device)
        with torch.no_grad():
            _ = model(input_ids=batch, use_cache=False)
        print(f"  [capture:{prefix}] batch {i}-{i+batch_size}/{enc.shape[0]} "
              f"in {time.time()-t0:.1f}s", flush=True)
    for h in handles:
        h.remove()
    return dict(counters)


def phase_capture(args: argparse.Namespace) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(args.base_dir), trust_remote_code=True)
    enc = _tokenize_calib(tok, args)

    # Explicit max_memory with per-GPU headroom. transformers 5.5's futures-based
    # loader packs `device_map="auto"` GPUs to ~full and leaves no activation
    # headroom — a 49 GB bf16 model across 2x24 GB OOMs at load (pod 37588132,
    # 2026-05-24). Capping each GPU below 24 GiB forces the overflow to CPU
    # (low_cpu_mem_usage + offload) deterministically, and still uses every GPU.
    max_memory = None
    if args.max_gpu_gib > 0:
        ngpu = torch.cuda.device_count()
        max_memory = {i: f"{args.max_gpu_gib}GiB" for i in range(ngpu)}
        max_memory["cpu"] = f"{args.max_cpu_gib}GiB"
        print(f"[capture] max_memory={max_memory}")

    meta = {
        "base_dir": str(args.base_dir), "variant_dir": str(args.variant_dir),
        "n_seq": args.n_seq, "seq_len": args.seq_len,
        "tokens_total": args.n_seq * args.seq_len, "prefixes": {},
    }
    for label, mdir, prefix in (("base (teacher)", args.base_dir, "h_base"),
                                ("variant (student)", args.variant_dir, "h_var")):
        print(f"\n[capture] loading {label} from {mdir} (BF16 device_map=auto)...")
        t0 = time.time()
        dmap = _resolve_device_map(Path(mdir), max_memory, torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            str(mdir), torch_dtype=torch.bfloat16, device_map=dmap,
            max_memory=(max_memory if dmap == "auto" else None),
            trust_remote_code=True, low_cpu_mem_usage=True)
        model.eval()
        print(f"[capture] loaded in {time.time()-t0:.0f}s")
        counters = _capture_one(model, enc, cache_dir, prefix, args.batch_size)
        meta["prefixes"][prefix] = {"model_dir": str(mdir),
                                    "n_batches_per_layer": counters}
        del model
        gc.collect()
        torch.cuda.empty_cache()

    with open(cache_dir / "capture_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    total_mb = sum(p.stat().st_size for p in cache_dir.glob("h_*.pt")) / 1e6
    print(f"\n[capture] DONE; cache {total_mb:.0f} MB at {cache_dir}")
    return 0


# ─── Phase 2: per-layer paired TopK-MSE calibration ──────────────────────────

def load_h_for_layer(cache_dir: Path, prefix: str, li: int) -> torch.Tensor:
    """Concatenate all per-batch chunks for (prefix, layer li) into (N, H)."""
    chunks = sorted(cache_dir.glob(f"{prefix}_layer_{li:02d}_batch_*.pt"))
    if not chunks:
        raise FileNotFoundError(f"no {prefix} chunks for layer {li} in {cache_dir}")
    parts = [torch.load(p, map_location="cpu") for p in chunks]
    return torch.cat(parts, dim=0)


def calibrate_one_layer(
    h_base_cpu: torch.Tensor,       # (N, H) BF16 CPU — teacher router input
    h_var_cpu: torch.Tensor,        # (N, H) BF16 CPU — student (variant) router input
    base_router: torch.Tensor,      # (128, H) FP32 CPU
    variant_router: torch.Tensor,   # (98, H) BF16 — to be optimized
    drop_map: List[int],            # 30 dropped 128e ids
    n_base: int,                    # 128
    calib_k: int,
    lr: float,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Returns the optimized variant_router (same shape, BF16 CPU).

    Teacher top-K + target logit values come from h_base @ base_W.T; the student
    optimizes variant_W so h_var @ variant_W.T matches those targets at the
    surviving mapped positions (paired per-token, FIX #3)."""
    n_keep = variant_router.shape[0]
    keep = [i for i in range(n_base) if i not in set(drop_map)]
    if len(keep) != n_keep:
        raise ValueError(f"keep {len(keep)} != variant rows {n_keep}")
    if h_base_cpu.shape[0] != h_var_cpu.shape[0]:
        raise ValueError(f"paired-row mismatch: h_base {h_base_cpu.shape[0]} "
                         f"!= h_var {h_var_cpu.shape[0]} (capture must use same tokens)")
    # 128-index → variant-index, -1 if dropped
    orig_to_var = torch.full((n_base,), -1, dtype=torch.long)
    for vi, oi in enumerate(keep):
        orig_to_var[oi] = vi

    hb = h_base_cpu.to(device=device, dtype=torch.float32, non_blocking=True)   # (N, H)
    hv = h_var_cpu.to(device=device, dtype=torch.float32, non_blocking=True)    # (N, H)
    teacher_W = base_router.to(device=device, dtype=torch.float32)              # (128, H)
    student_W = variant_router.detach().clone().to(
        device=device, dtype=torch.float32).requires_grad_(True)               # (98, H)
    orig_to_var = orig_to_var.to(device)

    # Teacher top-K (selection) + target logit values, computed ONCE on base input
    with torch.no_grad():
        teacher_logits = hb @ teacher_W.T                 # (N, 128)
        teacher_topk_vals, teacher_topk_idx = teacher_logits.topk(
            calib_k, dim=-1, sorted=False)                # (N, K), (N, K)
        mapped_idx = orig_to_var[teacher_topk_idx]        # (N, K), -1 if dropped
        valid_mask = mapped_idx >= 0                      # (N, K) bool
        mapped_idx_safe = mapped_idx.clamp(min=0)
        teacher_targets = teacher_topk_vals
        del teacher_logits

    n_valid_total = valid_mask.sum().item()
    pct_valid = 100.0 * n_valid_total / valid_mask.numel()
    print(f"    surviving teacher top-{calib_k} positions: {pct_valid:.1f}% "
          f"({n_valid_total:,}/{valid_mask.numel():,})")

    opt = torch.optim.AdamW([student_W], lr=lr, weight_decay=0.0)
    N = hv.shape[0]
    losses = []
    for step in range(steps):
        idx = torch.randint(0, N, (batch_size,), device=device)
        hv_b = hv[idx]                                  # (B, H) — VARIANT input
        tgt_b = teacher_targets[idx]                    # (B, K)
        mapped_b = mapped_idx_safe[idx]                 # (B, K)
        mask_b = valid_mask[idx].float()                # (B, K)

        student_logits = hv_b @ student_W.T             # (B, 98)
        student_at_pos = student_logits.gather(1, mapped_b)  # (B, K)

        diff_sq = (student_at_pos - tgt_b).pow(2) * mask_b
        denom = mask_b.sum().clamp(min=1.0)
        loss = diff_sq.sum() / denom

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if step % 20 == 0 or step == steps - 1:
            print(f"    step {step+1:3d}/{steps}: loss={loss.item():.6f} "
                  f"(initial {losses[0]:.6f})", flush=True)

    out = student_W.detach().to(dtype=variant_router.dtype, device="cpu")
    del hb, hv, teacher_W, student_W, teacher_targets, valid_mask, mapped_idx, mapped_idx_safe
    gc.collect()
    torch.cuda.empty_cache()
    return out


def phase_calibrate(args: argparse.Namespace) -> int:
    base_dir = Path(args.base_dir)
    var_dir = Path(args.variant_dir)
    cache_dir = Path(args.cache_dir)
    device = torch.device(args.device)

    meta_path = cache_dir / "capture_meta.json"
    if not meta_path.exists():
        print(f"FAIL: {meta_path} not found. Run --phase capture first.")
        return 1
    cap_meta = json.load(open(meta_path))
    if not args.restore:
        have = set(cap_meta.get("prefixes", {}))
        if {"h_base", "h_var"} - have:
            print(f"FAIL: capture cache missing prefixes {{'h_base','h_var'}} - {have}. "
                  f"Re-run --phase capture (paired base+variant required).")
            return 1

    # --restore reverts shards from backups and never consults the drop-map;
    # only read it on a real calibrate (else --restore crashes with no --drop-map).
    drop_map: Dict[int, List[int]] = {}
    if not args.restore:
        with open(args.drop_map) as f:
            drop_map = {int(k): [int(x) for x in v] for k, v in json.load(f).items()}

    var_routers = find_router_proj_tensors(var_dir)
    n_layers = len(var_routers)
    if not n_layers:
        print(f"FAIL: no router.proj.weight in {var_dir}")
        return 1
    print(f"[calibrate] layers to process: {n_layers}")

    print("[calibrate] extracting base router weights...")
    base_routers = load_base_router_weights(base_dir)
    n_base = next(iter(base_routers.values())).shape[0]
    print(f"[calibrate] base n_experts = {n_base}")

    shards: Dict[str, List[int]] = {}
    for li, (_, shard) in var_routers.items():
        shards.setdefault(shard, []).append(li)

    if args.restore:
        restored = 0
        for shard in shards:
            bak = var_dir / (shard + BACKUP_SUFFIX)
            if not bak.exists():
                print(f"WARN: {bak} missing")
                continue
            (var_dir / shard).write_bytes(bak.read_bytes())
            restored += 1
            print(f"  restored {shard}")
        print(f"OK restored {restored} shard(s)")
        return 0

    for shard in shards:
        bak = var_dir / (shard + BACKUP_SUFFIX)
        if not bak.exists():
            print(f"[calibrate] backup {shard} -> {bak.name}")
            bak.write_bytes((var_dir / shard).read_bytes())

    for shard, layer_ids in shards.items():
        shard_path = var_dir / shard
        print(f"\n[calibrate] === shard {shard} ({len(layer_ids)} layers) ===")
        meta = {}
        var_tensors: Dict[str, torch.Tensor] = {}
        with safe_open(str(shard_path), framework="pt") as f:
            meta = dict(f.metadata() or {})
            for k in f.keys():
                var_tensors[k] = f.get_tensor(k)

        for li in sorted(layer_ids):
            tname = var_routers[li][0]
            print(f"\n  layer {li}: optimizing {tname}")
            if li not in base_routers:
                print(f"    SKIP — base router for layer {li} not found")
                continue
            try:
                h_base = load_h_for_layer(cache_dir, "h_base", li)
                h_var = load_h_for_layer(cache_dir, "h_var", li)
            except FileNotFoundError as ex:
                print(f"    SKIP — {ex}")
                continue
            variant_W = var_tensors[tname]
            drops = drop_map.get(li, [])
            print(f"    h_base {tuple(h_base.shape)} h_var {tuple(h_var.shape)} "
                  f"variant {tuple(variant_W.shape)} dropped {len(drops)}")
            t0 = time.time()
            new_W = calibrate_one_layer(
                h_base_cpu=h_base,
                h_var_cpu=h_var,
                base_router=base_routers[li],
                variant_router=variant_W,
                drop_map=drops,
                n_base=n_base,
                calib_k=args.calib_k,
                lr=args.lr,
                steps=args.steps,
                batch_size=args.batch_size,
                device=device,
            )
            var_tensors[tname] = new_W
            print(f"    layer {li} done in {time.time()-t0:.0f}s")

        print(f"[calibrate] writing {shard_path}...")
        # ATOMIC WRITE via .tmp + replace — breaks any hardlinks the existing
        # shard has to sibling dirs. Origin: 2026-05-28 EAC arc on bs2 — A2_EAC
        # was created with `cp -al A2 A2_EAC` (hardlinks), so an in-place
        # save_file() wrote *through* the hardlink and modified A2 as well;
        # post-EAC A2 == A2_EAC byte-identical, EAC effectively a no-op for
        # variant separation. See feedback_canary_script_bug_invalidated_eac_arc.md.
        tmp_path = shard_path.with_suffix(shard_path.suffix + ".eac_write_tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        save_file(var_tensors, str(tmp_path), metadata=meta or None)
        try:
            nlink_before = shard_path.stat().st_nlink
        except OSError:
            nlink_before = -1
        tmp_path.replace(shard_path)
        if nlink_before > 1:
            print(f"  [defensive] broke hardlink on {shard_path.name} "
                  f"(was st_nlink={nlink_before}, now isolated)")
        print(f"  rewrote {shard_path.name}")

    print("\n[calibrate] DONE")
    return 0


# ─── Entry ───────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["capture", "calibrate", "both"],
                    required=True)

    ap.add_argument("--base-dir", required=True,
                    help="Base unpruned 128e model dir (teacher, HF BF16)")
    ap.add_argument("--variant-dir",
                    help="Pruned variant dir (student) — edited in calibrate; "
                         "forwarded in capture. Required for capture + calibrate.")
    ap.add_argument("--cache-dir",
                    default="/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/eac_cache",
                    help="Where to stream captured h tensors (persistent disk, NEVER /tmp)")
    ap.add_argument("--n-seq", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Capture batch size (small to fit 26B BF16 + offload)")
    ap.add_argument("--corpus-file", default=None,
                    help="Plain-text corpus to tokenize instead of WikiText-2.")
    ap.add_argument("--max-gpu-gib", type=float, default=20.0,
                    help="Per-GPU memory cap (GiB) for the capture model load. "
                         "<24 forces overflow to CPU so a 49GB bf16 model fits on "
                         "2x24GB without the auto-pack OOM. 0 = unbounded auto.")
    ap.add_argument("--max-cpu-gib", type=float, default=400.0,
                    help="CPU RAM budget (GiB) for offloaded layers during capture.")

    ap.add_argument("--drop-map",
                    help="JSON {layer_str: [dropped_expert_ids]} (calibrate)")
    ap.add_argument("--calib-k", "--top-k", dest="calib_k", type=int, default=16,
                    help="K for the TopK-MSE calibration window. Paper uses K WIDER "
                         "than native top-k (Deepseek 6/64 → 16); Gemma native top-k "
                         "is 8, so 16-24 recommended. NOT the model's native top-k.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--opt-batch-size", type=int, default=16384,
                    dest="opt_batch_size",
                    help="Token-batch size for the calibrate optimizer")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--restore", action="store_true",
                    help="Restore variant shards from .pre_eac_calibrate backups")

    args = ap.parse_args()
    capture_bs = args.batch_size  # preserve before the calibrate rebind below

    if args.phase in ("capture", "both") and not args.variant_dir:
        print("FAIL: --variant-dir required for capture (paired base+variant forward)")
        return 1
    if args.phase in ("calibrate", "both") and not args.restore:
        if not args.variant_dir or not args.drop_map:
            print("FAIL: --variant-dir and --drop-map required for calibrate")
            return 1

    if args.phase == "capture":
        return phase_capture(args)
    if args.phase == "calibrate":
        args.batch_size = args.opt_batch_size  # calibrate uses opt-batch-size
        return phase_calibrate(args)
    if args.phase == "both":
        args.batch_size = capture_bs
        rc = phase_capture(args)
        if rc != 0:
            return rc
        args.batch_size = args.opt_batch_size
        return phase_calibrate(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
