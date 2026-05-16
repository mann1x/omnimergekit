#!/usr/bin/env python3
"""router_soft_transfer.py — T18 Step 1c: distribute dropped experts' router
mass onto cosine-similar surviving experts.

When `expert_drop.py` slices the router from 128 → 98 rows, the mass that
USED to flow to the 30 dropped experts is gone — at inference, the top-8
softmax over surviving 98 redistributes their would-be share by chance.
This is the off-manifold routing pathology behind the rumination.

The fix (inverse of "Diversifying Expert Knowledge" merge, ACL Findings 2025):
for each dropped expert d in the BASE 128-router, find the top-k cosine-
similar SURVIVING expert rows in the same base router. Add a weighted
fraction of d's row to those survivors' rows in the VARIANT router. So at
inference, hidden states that would have lit up d now (partially) excite
its closest cousins instead.

Three router tensors per layer in Gemma 4 NVFP4A16:
  router.proj.weight         (98, 2816)  ← we edit this
  router.scale               (2816,)     ← untouched
  router.per_expert_scale    (98,)       ← untouched
(scale + per_expert_scale untouched at Step 1 — Step 2 EAC-MoE jointly
optimizes them.)

Usage:
    python scripts/router_soft_transfer.py \
        --base-dir   google/gemma-4-26B-A4B-it \
        --variant-dir google/gemma-4-A4B-98e-v5fixed-sweep-A2_lp4_uni-NVFP4A16 \
        --drop-map   scripts/v5fixed_sweep_A2_lp4_uni_drop_map.json \
        --alpha 0.3 \
        --top-k 3

α: fraction of d's row injected per neighbor (split top-k equally by cos sim).
top-k: how many neighbors absorb each dropped row.

Backup created at <shard>.pre_soft_transfer before edit. --restore undoes it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BACKUP_SUFFIX = ".pre_soft_transfer"


def find_router_proj_tensors(model_dir: Path) -> dict[int, tuple[str, str]]:
    """Returns {layer_idx: (tensor_name, shard_filename)} for router.proj.weight."""
    idx_path = model_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        raise SystemExit(f"FAIL: {idx_path} not found")
    with open(idx_path) as f:
        wm = json.load(f)["weight_map"]
    out = {}
    for k, shard in wm.items():
        if k.endswith("router.proj.weight"):
            # Find "layers.N." segment
            m = [seg for seg in k.split(".") if seg.isdigit()]
            if not m:
                continue
            # last digit token == layer
            li = int(m[-1])
            out[li] = (k, shard)
    return out


def load_router_proj(model_dir: Path, tensor_name: str, shard: str) -> torch.Tensor:
    with safe_open(str(model_dir / shard), framework="pt") as f:
        return f.get_tensor(tensor_name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", required=True,
                    help="Base unpruned model dir, e.g. google/gemma-4-26B-A4B-it")
    ap.add_argument("--variant-dir", required=True,
                    help="Pruned NVFP4A16 variant dir to edit")
    ap.add_argument("--drop-map", required=True,
                    help="JSON {layer_str: [dropped_expert_ids]} produced by "
                         "generate_drop_map_v5.py — needed to map variant indices "
                         "back to original 128e indices.")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="Total fraction of dropped d's row injected (split across "
                         "top-k neighbors weighted by cos sim). Try 0.1, 0.3, 0.5.")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Number of cosine-similar surviving neighbors per drop")
    ap.add_argument("--restore", action="store_true",
                    help="Restore variant shards from .pre_soft_transfer backups")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print per-layer plan, don't write")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    var_dir = Path(args.variant_dir)

    # Locate router.proj.weight in both
    var_routers = find_router_proj_tensors(var_dir)
    if not var_routers:
        print(f"FAIL: no router.proj.weight tensors in {var_dir}")
        return 1

    # Group variant shards (one backup per shard, possibly multiple layers per shard)
    shards: dict[str, list[int]] = {}
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

    with open(args.drop_map) as f:
        drop_map = {int(k): [int(x) for x in v] for k, v in json.load(f).items()}

    # Sanity: drop_map references 128e expert ids. Variant rows are 0..97 (98 surviving).
    # Mapping convention used by expert_drop.py: surviving expert IDs are sorted by
    # original 128e id with gaps removed → variant row r corresponds to original
    # surviving-id `keep[r]` where keep = sorted(set(0..127) - dropped).
    # We use this convention below.

    base_routers = find_router_proj_tensors(base_dir)
    if not base_routers:
        print(f"FAIL: no router.proj.weight tensors in {base_dir}")
        return 1
    n_layers = len(var_routers)
    if len(base_routers) != n_layers:
        print(f"WARN: base has {len(base_routers)} layers, variant has {n_layers}")

    print(f"layers to process: {n_layers}")
    print(f"α={args.alpha} top-k={args.top_k}")
    print(f"variant shards to edit: {len(shards)} ({list(shards)})")

    # Read base router rows once per shard (lots of tensors but small)
    # Process shard-by-shard for memory efficiency.
    for shard, layer_ids in shards.items():
        shard_path = var_dir / shard
        bak_path = var_dir / (shard + BACKUP_SUFFIX)
        if not args.dry_run and not bak_path.exists():
            bak_path.write_bytes(shard_path.read_bytes())
            print(f"  backup {shard} -> {bak_path.name}")

        # Load all tensors from variant shard
        meta = {}
        var_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(shard_path), framework="pt") as f:
            meta = dict(f.metadata() or {})
            for k in f.keys():
                var_tensors[k] = f.get_tensor(k)

        for li in sorted(layer_ids):
            var_name = var_routers[li][0]
            var_router = var_tensors[var_name]  # (98, hidden_size), BF16
            n_keep, hidden = var_router.shape

            # Load corresponding base router (128, hidden_size)
            base_name, base_shard = base_routers[li]
            base_router = load_router_proj(base_dir, base_name, base_shard).to(torch.float32)
            n_base = base_router.shape[0]

            dropped = sorted(drop_map.get(li, []))
            keep = [i for i in range(n_base) if i not in set(dropped)]
            if len(keep) != n_keep:
                print(f"  L{li}: keep={len(keep)} != variant rows={n_keep}; skipping")
                continue
            keep_map_var_to_orig = {r: keep[r] for r in range(n_keep)}
            keep_map_orig_to_var = {v: k for k, v in keep_map_var_to_orig.items()}

            # Normalize base rows for cosine similarity
            row_norms = base_router.norm(dim=1, keepdim=True).clamp(min=1e-12)
            base_normed = base_router / row_norms

            # Edit accumulator (FP32 for stability), starts from variant router
            var_router_fp32 = var_router.to(torch.float32).clone()

            transfers_logged = 0
            for d in dropped:
                d_vec = base_normed[d]  # (hidden,)
                # cos sim with every other row
                sims = base_normed @ d_vec  # (n_base,)
                sims[d] = -2.0  # exclude self
                # Mask non-survivors
                for x in dropped:
                    sims[x] = -2.0
                # Take top-k survivors
                topk = torch.topk(sims, k=args.top_k)
                top_orig_ids = topk.indices.tolist()
                top_sims = topk.values.clamp(min=0.0)  # neg sims should not contribute
                if top_sims.sum() <= 0:
                    continue
                weights = (top_sims / top_sims.sum()).tolist()
                for orig_neighbor, w in zip(top_orig_ids, weights):
                    var_row = keep_map_orig_to_var.get(orig_neighbor)
                    if var_row is None:
                        continue
                    inject = args.alpha * w * base_router[d]
                    var_router_fp32[var_row] = var_router_fp32[var_row] + inject
                    transfers_logged += 1

            if args.dry_run:
                print(f"  L{li}: would transfer {transfers_logged} rows "
                      f"({len(dropped)} dropped × top-{args.top_k})")
            else:
                var_tensors[var_name] = var_router_fp32.to(var_router.dtype)
                if li % 5 == 0:
                    print(f"  L{li}: transferred (drops={len(dropped)} sum-α={args.alpha})")

        if not args.dry_run:
            save_file(var_tensors, str(shard_path), metadata=meta or None)
            print(f"  rewrote {shard}")

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
