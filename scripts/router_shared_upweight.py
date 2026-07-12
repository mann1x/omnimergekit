#!/usr/bin/env python3
"""router_shared_upweight.py — T18 Step 1b: dial up the always-on shared FFN.

Gemma 4 26B-A4B has a parallel dense `mlp.*` block at every MoE layer that
runs alongside the routed expert mixture. After expert drop, the routed
mixture is unreliable on hard prompts (rumination); leaning more on the
shared FFN (α > 1) can mask the routing damage.

The mlp tensors are NVFP4A16 (4-bit). We do NOT touch the 4-bit weights —
instead scale the FP32 `weight_scale_2` of `mlp.down_proj` by α. Since the
dequantized weight = (qweight × weight_scale) × weight_scale_2, multiplying
weight_scale_2 by α multiplies the layer's shared output by α. Same effect
as a clean re-quant at α-scaled weights, without re-quantizing.

We only scale `down_proj.weight_scale_2` (the OUTPUT projection of the FFN)
to avoid double-counting the α gain through both gate and up branches.

Usage:
    python scripts/router_shared_upweight.py \
        --model-dir google/gemma-4-A4B-98e-v5fixed-sweep-A2_lp4_uni-NVFP4A16 \
        --alpha 1.2
    python scripts/router_shared_upweight.py --model-dir <dir> --restore   # undo

Reversible: backs up edited shards once, restore touches `down_proj.weight_scale_2`
on every layer back to the saved values.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BACKUP_SUFFIX = ".pre_shared_upweight"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--alpha", type=float, default=None,
                    help="Scale factor on shared mlp down_proj output, e.g. 1.2")
    ap.add_argument("--restore", action="store_true",
                    help="Restore shards from .pre_shared_upweight backups (undo)")
    ap.add_argument("--target", default="mlp.down_proj.weight_scale_2",
                    help="Tensor name suffix to scale (default: down_proj.weight_scale_2)")
    args = ap.parse_args()

    d = Path(args.model_dir)
    idx_path = d / "model.safetensors.index.json"
    if not idx_path.exists():
        print(f"FAIL: {idx_path} not found")
        return 1

    with open(idx_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    # Collect target tensors (one per layer)
    targets = [k for k in weight_map if k.endswith(args.target)]
    if not targets:
        # Maybe model uses unsuffixed scale; try `weight_scale` if user passed it
        print(f"FAIL: no tensors found matching suffix '{args.target}'. "
              f"Inspect {idx_path}.")
        return 2
    targets.sort()

    # Group by shard
    shards: dict[str, list[str]] = {}
    for k in targets:
        shards.setdefault(weight_map[k], []).append(k)

    if args.restore:
        # Restore each affected shard from its .pre_shared_upweight backup
        restored = 0
        for shard in shards:
            bak = d / (shard + BACKUP_SUFFIX)
            if not bak.exists():
                print(f"WARN: {bak} missing; skipping {shard}")
                continue
            # Backup -> live
            (d / shard).write_bytes(bak.read_bytes())
            restored += 1
            print(f"  restored {shard}")
        print(f"OK restored {restored} shard(s)")
        return 0

    if args.alpha is None:
        print("FAIL: --alpha required (unless --restore)")
        return 1
    if args.alpha <= 0:
        print(f"FAIL: --alpha must be > 0 (got {args.alpha})")
        return 1

    print(f"target tensors: {len(targets)} (one per layer expected)")
    print(f"alpha: {args.alpha}")
    print(f"shards to edit: {len(shards)}")

    edited_count = 0
    for shard, keys in shards.items():
        shard_path = d / shard
        bak_path = d / (shard + BACKUP_SUFFIX)
        # One-time backup
        if not bak_path.exists():
            bak_path.write_bytes(shard_path.read_bytes())
            print(f"  backup {shard} -> {bak_path.name}")
        # Load all tensors from this shard
        tensors: dict[str, torch.Tensor] = {}
        meta = {}
        with safe_open(str(shard_path), framework="pt") as f:
            meta = dict(f.metadata() or {})
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        # Apply alpha to the target tensors only
        for k in keys:
            t = tensors[k]
            orig_dtype = t.dtype
            tensors[k] = (t.to(torch.float32) * args.alpha).to(orig_dtype)
            edited_count += 1
        save_file(tensors, str(shard_path), metadata=meta or None)
        print(f"  rewrote {shard} ({len(keys)} tensor(s) scaled by α)")

    print(f"OK scaled {edited_count} tensor(s) by α={args.alpha}")
    print("   restore with: --restore")
    return 0


if __name__ == "__main__":
    sys.exit(main())
