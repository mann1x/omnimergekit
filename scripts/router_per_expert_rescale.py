#!/usr/bin/env python3
"""router_per_expert_rescale.py — T18 mixing-magnitude probe (routed experts).

Sibling of router_shared_upweight.py (which scales the SHARED FFN path). This
one scales the ROUTED experts' post-selection mixing weights by editing
`router.per_expert_scale` (a per-expert bf16 vector, one per MoE layer).

WHY (council csl-2026-05-24-0949-41ea): EAC-MoE TopK-MSE calibrates which
experts are SELECTED, but `per_expert_scale` is applied AFTER top-k/renorm
(Gemma4TextRouter: `top_k_weights = renorm(softmax(proj(...))) * per_expert_scale`),
so it sets the MIXING MAGNITUDE of the routed contribution, which EAC never
touches. If rumination/looping is driven by off-manifold mixing magnitude
(the 98e model leaning too hard / too little on the routed mixture vs the 128e
manifold) rather than by wrong expert selection, THIS is the lever — and it is
a free knob NOT in the exhausted set (top-k dial / shared-α / soft-transfer).

This is the FAST probe: a pure weight-edit (no forward pass), global α across
all layers. Sweep α both directions (e.g. 0.85 0.90 0.95 1.05 1.10) and re-run
the 5-doc canary set (scripts/ifeval_rumination_canaries.json) to see if any α
pulls the looping prompts into the 128e-clean band. If one does, the
mixing-magnitude hypothesis is confirmed and the principled per-layer
data-driven recalibration (renorm-denominator ratio vs 128e, or full Router-KD
which trains per_expert_scale too) is justified.

Reversible: --out-dir copies the source then edits the copy (default, never
clobbers the champion); without --out-dir it edits in place with a
<shard>.pre_per_expert_rescale backup. --restore reverts in-place edits.

Usage:
    # non-destructive probe on a copy:
    python scripts/router_per_expert_rescale.py \
        --model-dir google/gemma-4-A4B-98e-v5-coder-it \
        --out-dir   google/gemma-4-A4B-98e-v5-coder-pes090-it --alpha 0.90
    # in-place + undo:
    python scripts/router_per_expert_rescale.py --model-dir <dir> --alpha 0.90
    python scripts/router_per_expert_rescale.py --model-dir <dir> --restore
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BACKUP_SUFFIX = ".pre_per_expert_rescale"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="Source pruned variant dir")
    ap.add_argument("--out-dir", default=None,
                    help="Copy source here and edit the copy (non-destructive). "
                         "If omitted, edits --model-dir in place with backup.")
    ap.add_argument("--alpha", type=float, default=None,
                    help="Global scale on router.per_expert_scale (e.g. 0.90)")
    ap.add_argument("--target", default="router.per_expert_scale",
                    help="Tensor name suffix to scale")
    ap.add_argument("--restore", action="store_true",
                    help="Restore in-place shards from .pre_per_expert_rescale backups")
    args = ap.parse_args()

    # Resolve the dir we actually edit
    src = Path(args.model_dir)
    if args.restore:
        edit_dir = src
    elif args.out_dir:
        edit_dir = Path(args.out_dir)
        if not edit_dir.exists():
            print(f"[copy] {src} -> {edit_dir}")
            shutil.copytree(src, edit_dir)
    else:
        edit_dir = src

    idx_path = edit_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        print(f"FAIL: {idx_path} not found")
        return 1
    weight_map = json.load(open(idx_path))["weight_map"]

    targets = sorted(k for k in weight_map if k.endswith(args.target))
    if not targets:
        print(f"FAIL: no tensors matching suffix '{args.target}' in {idx_path}")
        return 2
    shards: dict[str, list[str]] = {}
    for k in targets:
        shards.setdefault(weight_map[k], []).append(k)

    if args.restore:
        restored = 0
        for shard in shards:
            bak = edit_dir / (shard + BACKUP_SUFFIX)
            if not bak.exists():
                print(f"WARN: {bak} missing; skipping {shard}")
                continue
            (edit_dir / shard).write_bytes(bak.read_bytes())
            restored += 1
            print(f"  restored {shard}")
        print(f"OK restored {restored} shard(s)")
        return 0

    if args.alpha is None or args.alpha <= 0:
        print(f"FAIL: --alpha must be > 0 (got {args.alpha})")
        return 1

    print(f"target tensors: {len(targets)} (one per MoE layer expected) "
          f"alpha={args.alpha} shards={len(shards)} edit_dir={edit_dir}")
    in_place = args.out_dir is None
    edited = 0
    for shard, keys in shards.items():
        shard_path = edit_dir / shard
        if in_place:
            bak = edit_dir / (shard + BACKUP_SUFFIX)
            if not bak.exists():
                bak.write_bytes(shard_path.read_bytes())
                print(f"  backup {shard} -> {bak.name}")
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(shard_path), framework="pt") as f:
            meta = dict(f.metadata() or {})
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        for k in keys:
            t = tensors[k]
            tensors[k] = (t.to(torch.float32) * args.alpha).to(t.dtype)
            edited += 1
        save_file(tensors, str(shard_path), metadata=meta or None)
        print(f"  rewrote {shard} ({len(keys)} tensor(s) ×{args.alpha})")

    print(f"OK scaled {edited} per_expert_scale tensor(s) by α={args.alpha} in {edit_dir}")
    if in_place:
        print("   undo with: --restore")
    return 0


if __name__ == "__main__":
    sys.exit(main())
