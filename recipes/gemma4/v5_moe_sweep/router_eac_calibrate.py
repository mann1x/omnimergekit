#!/usr/bin/env python3
"""router_eac_calibrate.py — T18 Step 2: EAC-MoE / QESC layer-by-layer
TopK-MSE router calibration on a pruned MoE variant.

Paper: "EAC-MoE: Expert-Selection Aware Compressor for MoE LLMs"
       (arXiv:2508.01625, ACL 2025).

Two phases, runnable separately so a single capture can serve many variants:

  --phase capture
      Forward base 128e on WikiText-2 calibration data, capture the hidden
      state arriving at each MoE layer's router (post-pre_feedforward_layernorm
      output) and stream to <cache-dir>/h_layer_NN_batch_MM.pt. ~30 min on a
      3090 with device_map=auto offload. Disk cost ~40 GB for 128 × 2048 tok.

  --phase calibrate
      Per-layer: load cached h, compute teacher's top-K logits via the BASE
      128-row router (extracted from base safetensors), keep only positions
      that survived pruning, optimize the VARIANT's router.proj.weight (BF16)
      via AdamW topk-MSE loss on those positions. Write back to variant
      safetensors. ~10-15 min on a 3090 per variant.

  --phase both
      capture, then calibrate.

Mechanics:
  - The teacher's role is filled by the unpruned 128e router weights ONLY
    (extracted from base safetensors, ~22 MB total). We do NOT need the
    base experts — the EAC-MoE loss is on routing logits, not expert outputs.
  - h comes from a forward pass of BASE 128e on calibration data. (Variant's
    actual runtime h diverges from base's, but is bounded and per-paper this
    is the standard calibration assumption.)
  - For each token, teacher's top-K positions in 128-space are mapped to
    variant-space via the drop map's `keep[]`. Positions that map to a
    DROPPED expert are skipped — we only match positions the variant has.
  - Variant router edits land on `router.proj.weight` (BF16). `router.scale`
    and `router.per_expert_scale` are NOT modified — Step 3 (Router KD) would
    jointly optimize them; Step 2 keeps surgery minimal.

Backup: each variant shard touched is copied to `<shard>.pre_eac_calibrate`
before edit. --restore reverts.

Usage:
    # Capture once (slow, ~30 min):
    python scripts/router_eac_calibrate.py --phase capture \\
        --base-dir google/gemma-4-26B-A4B-it \\
        --cache-dir /tmp/eac_cache \\
        --n-seq 128 --seq-len 2048

    # Then calibrate per variant (~10-15 min each):
    python scripts/router_eac_calibrate.py --phase calibrate \\
        --base-dir google/gemma-4-26B-A4B-it \\
        --variant-dir google/gemma-4-A4B-98e-v5fixed-sweep-A2_lp4_uni-NVFP4A16 \\
        --drop-map  scripts/v5fixed_sweep_A2_lp4_uni_drop_map.json \\
        --cache-dir /tmp/eac_cache \\
        --top-k 8 --lr 1e-3 --steps 100 --batch-size 16384

NB: cache-dir is reusable across variants (same base, same calib data → same h).
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


# ─── Phase 1: capture h's via forward hook on base 128e ──────────────────────

def phase_capture(args: argparse.Namespace) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(args.base_dir)
    print(f"[capture] loading base from {base_dir} (BF16 device_map=auto)...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[capture] loaded in {time.time()-t0:.0f}s")

    # Locate pre_feedforward_layernorm modules — its output is the router input
    # We exclude "_2" variant which is the shared-mlp's input layernorm.
    hooked: Dict[int, int] = {}  # layer_idx -> batch counter
    batch_counters: Dict[int, int] = {}

    def make_hook(li: int):
        def hook(module, _input, output):
            # output: (B, T, H). Stream to disk per batch.
            bi = batch_counters[li]
            path = cache_dir / f"h_layer_{li:02d}_batch_{bi:04d}.pt"
            t = output.detach().cpu().to(torch.bfloat16).reshape(-1, output.shape[-1]).clone()
            torch.save(t, path)
            batch_counters[li] = bi + 1
        return hook

    n_layers_seen = 0
    for name, mod in model.named_modules():
        # Match exactly `....layers.N.pre_feedforward_layernorm` (NOT `_2`)
        if name.endswith(".pre_feedforward_layernorm"):
            digits = [int(p) for p in name.split(".") if p.isdigit()]
            if not digits:
                continue
            li = digits[-1]
            mod.register_forward_hook(make_hook(li))
            hooked[li] = 0
            batch_counters[li] = 0
            n_layers_seen += 1
    print(f"[capture] hooks installed on {n_layers_seen} layers: {sorted(hooked)[:5]} ... {sorted(hooked)[-5:]}")
    if n_layers_seen == 0:
        print("FAIL: no pre_feedforward_layernorm modules found; check architecture")
        return 1

    # Load calibration data — either WikiText (default) or a corpus text file
    if args.corpus_file:
        cpath = Path(args.corpus_file)
        if not cpath.exists():
            print(f"FAIL: --corpus-file {cpath} not found")
            return 1
        print(f"[capture] loading task corpus {cpath} ({cpath.stat().st_size/1e6:.1f} MB)")
        with open(cpath) as f:
            text = f.read()
    else:
        print("[capture] loading WikiText-2 raw (train split)...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(s for s in ds["text"] if s.strip())
    print(f"[capture] tokenizing {len(text):,} chars...")
    enc = tok(text, return_tensors="pt", truncation=False)["input_ids"][0]
    needed = args.n_seq * args.seq_len
    if enc.numel() < needed:
        print(f"WARN: only {enc.numel()} tokens, need {needed}; reducing n_seq")
        args.n_seq = enc.numel() // args.seq_len
    enc = enc[:args.n_seq * args.seq_len].view(args.n_seq, args.seq_len)
    print(f"[capture] forwarding {args.n_seq} seq × {args.seq_len} tok = {args.n_seq * args.seq_len:,} tokens")

    device = next(model.parameters()).device
    for i in range(0, args.n_seq, args.batch_size):
        t0 = time.time()
        batch = enc[i:i + args.batch_size].to(device)
        with torch.no_grad():
            _ = model(input_ids=batch, use_cache=False)
        print(f"  [capture] batch {i}-{i+args.batch_size}/{args.n_seq} "
              f"in {time.time()-t0:.1f}s", flush=True)

    print("[capture] saving capture metadata...")
    meta = {
        "base_dir": str(base_dir),
        "n_seq": args.n_seq,
        "seq_len": args.seq_len,
        "n_layers": n_layers_seen,
        "n_batches_per_layer": batch_counters,
        "tokens_total": args.n_seq * args.seq_len,
    }
    with open(cache_dir / "capture_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    total_size_mb = sum(p.stat().st_size for p in cache_dir.glob("h_*.pt")) / 1e6
    print(f"[capture] DONE; cache size {total_size_mb:.0f} MB at {cache_dir}")
    return 0


# ─── Phase 2: per-layer TopK-MSE calibration ─────────────────────────────────

def load_h_for_layer(cache_dir: Path, li: int) -> torch.Tensor:
    """Concatenate all per-batch chunks for layer li into one tensor (N, H)."""
    chunks = sorted(cache_dir.glob(f"h_layer_{li:02d}_batch_*.pt"))
    if not chunks:
        raise FileNotFoundError(f"no chunks for layer {li} in {cache_dir}")
    parts = [torch.load(p, map_location="cpu") for p in chunks]
    return torch.cat(parts, dim=0)


def calibrate_one_layer(
    h_cpu: torch.Tensor,            # (N, H) BF16 CPU
    base_router: torch.Tensor,      # (128, H) FP32 CPU
    variant_router: torch.Tensor,   # (98, H) BF16 — to be optimized
    drop_map: List[int],            # 30 dropped 128e ids
    n_base: int,                    # 128
    top_k: int,
    lr: float,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Returns the optimized variant_router (same shape, BF16 CPU)."""
    n_keep = variant_router.shape[0]
    keep = [i for i in range(n_base) if i not in set(drop_map)]
    if len(keep) != n_keep:
        raise ValueError(f"keep {len(keep)} != variant rows {n_keep}")
    # 128-index → variant-index, -1 if dropped
    orig_to_var = torch.full((n_base,), -1, dtype=torch.long)
    for vi, oi in enumerate(keep):
        orig_to_var[oi] = vi

    # Move tensors to GPU as FP32 for the optimization (small enough)
    h = h_cpu.to(device=device, dtype=torch.float32, non_blocking=True)  # (N, H)
    teacher_W = base_router.to(device=device, dtype=torch.float32)        # (128, H)
    student_W = variant_router.detach().clone().to(
        device=device, dtype=torch.float32).requires_grad_(True)         # (98, H)
    orig_to_var = orig_to_var.to(device)

    # Compute teacher's top-K per token, ONCE (no_grad)
    with torch.no_grad():
        teacher_logits = h @ teacher_W.T                  # (N, 128)
        teacher_topk_vals, teacher_topk_idx = teacher_logits.topk(
            top_k, dim=-1, sorted=False)                  # (N, K), (N, K)
        # Filter: mask out dropped positions
        mapped_idx = orig_to_var[teacher_topk_idx]        # (N, K), -1 if dropped
        valid_mask = mapped_idx >= 0                      # (N, K) bool
        # For dropped positions, set mapped_idx to 0 (we'll mask the loss)
        mapped_idx_safe = mapped_idx.clamp(min=0)
        teacher_targets = teacher_topk_vals
        del teacher_logits

    # Per-token surviving K count (useful for normalization)
    n_valid_total = valid_mask.sum().item()
    pct_valid = 100.0 * n_valid_total / valid_mask.numel()
    print(f"    surviving teacher top-K positions: {pct_valid:.1f}% "
          f"({n_valid_total:,}/{valid_mask.numel():,})")

    # Optimizer
    opt = torch.optim.AdamW([student_W], lr=lr, weight_decay=0.0)
    N = h.shape[0]
    losses = []
    for step in range(steps):
        # Random batch indices
        idx = torch.randint(0, N, (batch_size,), device=device)
        h_b = h[idx]                                    # (B, H)
        tgt_b = teacher_targets[idx]                    # (B, K)
        mapped_b = mapped_idx_safe[idx]                 # (B, K)
        mask_b = valid_mask[idx].float()                # (B, K)

        # Student logits at the mapped positions
        # student_W: (98, H); h_b @ student_W.T: (B, 98); gather at mapped_b
        student_logits = h_b @ student_W.T              # (B, 98)
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
    del h, teacher_W, student_W, teacher_targets, valid_mask, mapped_idx, mapped_idx_safe
    gc.collect()
    torch.cuda.empty_cache()
    return out


def phase_calibrate(args: argparse.Namespace) -> int:
    base_dir = Path(args.base_dir)
    var_dir = Path(args.variant_dir)
    cache_dir = Path(args.cache_dir)
    device = torch.device(args.device)

    # Sanity: capture metadata present
    meta_path = cache_dir / "capture_meta.json"
    if not meta_path.exists():
        print(f"FAIL: {meta_path} not found. Run --phase capture first.")
        return 1

    # Load drop map
    with open(args.drop_map) as f:
        drop_map = {int(k): [int(x) for x in v] for k, v in json.load(f).items()}

    # Locate routers
    var_routers = find_router_proj_tensors(var_dir)
    n_layers = len(var_routers)
    if not n_layers:
        print(f"FAIL: no router.proj.weight in {var_dir}")
        return 1
    print(f"[calibrate] layers to process: {n_layers}")

    # Load base router weights into RAM (small — ~22 MB total)
    print("[calibrate] extracting base router weights...")
    base_routers = load_base_router_weights(base_dir)
    n_base = next(iter(base_routers.values())).shape[0]
    print(f"[calibrate] base n_experts = {n_base}")

    # Group variant shards (for batched write-back + single backup per shard)
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

    # Backups
    for shard in shards:
        bak = var_dir / (shard + BACKUP_SUFFIX)
        if not bak.exists():
            print(f"[calibrate] backup {shard} -> {bak.name}")
            bak.write_bytes((var_dir / shard).read_bytes())

    # Calibrate each layer; accumulate edits per-shard so we save once per shard
    for shard, layer_ids in shards.items():
        shard_path = var_dir / shard
        print(f"\n[calibrate] === shard {shard} ({len(layer_ids)} layers) ===")
        # Load entire shard
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
                h = load_h_for_layer(cache_dir, li)
            except FileNotFoundError as ex:
                print(f"    SKIP — {ex}")
                continue
            variant_W = var_tensors[tname]
            drops = drop_map.get(li, [])
            print(f"    h shape {tuple(h.shape)} "
                  f"variant {tuple(variant_W.shape)} dropped {len(drops)}")
            t0 = time.time()
            new_W = calibrate_one_layer(
                h_cpu=h,
                base_router=base_routers[li],
                variant_router=variant_W,
                drop_map=drops,
                n_base=n_base,
                top_k=args.top_k,
                lr=args.lr,
                steps=args.steps,
                batch_size=args.batch_size,
                device=device,
            )
            var_tensors[tname] = new_W
            print(f"    layer {li} done in {time.time()-t0:.0f}s")

        # Save back
        print(f"[calibrate] writing {shard_path}...")
        save_file(var_tensors, str(shard_path), metadata=meta or None)
        print(f"  rewrote {shard_path.name}")

    print("\n[calibrate] DONE")
    return 0


# ─── Entry ───────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["capture", "calibrate", "both"],
                    required=True)

    # Capture-side
    ap.add_argument("--base-dir", required=True,
                    help="Base unpruned model dir (HF format, BF16)")
    ap.add_argument("--cache-dir", default="/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/eac_cache",
                    help="Where to stream captured h tensors")
    ap.add_argument("--n-seq", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Capture batch size (small to fit 26B BF16 on 24 GB)")
    ap.add_argument("--corpus-file", default=None,
                    help="Path to a plain text corpus to tokenize instead of "
                         "WikiText-2. Use scripts/build_diff_corpus.py to assemble "
                         "a task-specific corpus from 128e samples.")

    # Calibrate-side
    ap.add_argument("--variant-dir",
                    help="Pruned NVFP4A16 variant dir to edit (calibrate phase)")
    ap.add_argument("--drop-map",
                    help="JSON {layer_str: [dropped_expert_ids]}")
    ap.add_argument("--top-k", type=int, default=8,
                    help="K in TopK-MSE; default 8 (Gemma 4 native top-k)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--opt-batch-size", type=int, default=16384,
                    dest="opt_batch_size",
                    help="Token-batch size for the calibrate optimizer")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--restore", action="store_true",
                    help="Restore variant shards from .pre_eac_calibrate backups")

    args = ap.parse_args()

    # Wire opt_batch_size → batch_size for calibrate (capture uses its own)
    if args.phase in ("calibrate", "both"):
        if not args.restore:
            if not args.variant_dir or not args.drop_map:
                print("FAIL: --variant-dir and --drop-map required for calibrate")
                return 1
        # Re-bind: calibrate's batch_size argv is opt-batch-size
        args.batch_size = args.opt_batch_size

    if args.phase == "capture":
        return phase_capture(args)
    if args.phase == "calibrate":
        return phase_calibrate(args)
    if args.phase == "both":
        rc = phase_capture(args)
        if rc != 0:
            return rc
        # Reset args.batch_size for calibrate
        args.batch_size = args.opt_batch_size
        return phase_calibrate(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
