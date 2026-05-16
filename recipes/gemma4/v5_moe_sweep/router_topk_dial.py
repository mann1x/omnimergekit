#!/usr/bin/env python3
"""router_topk_dial.py — T18 Step 1a: edit Gemma 4 MoE top-k in place.

Why: after expert drop, top-k stays at 8 of (now) 98 experts → routing
density jumps from 8/128=6.25% to 8/98=8.16%, 30% denser than pretrain.
Dropping to top-6 preserves the original density (6/98=6.12%).

Edits config.json `top_k_experts` (Gemma 4 non-standard key). vLLM reads
at boot, no re-quant needed. Reversible — keep a backup of config.json.

Usage:
    python scripts/router_topk_dial.py \
        --model-dir google/gemma-4-A4B-98e-v5fixed-sweep-A2_lp4_uni-NVFP4A16 \
        --top-k 6
    python scripts/router_topk_dial.py --model-dir <dir> --restore   # undo

Exit 0 = success, 1 = config or model dir not found, 2 = key not found.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


# Gemma 4 26B-A4B uses these non-standard MoE keys (see config.json text_config).
TOPK_KEYS = ("top_k_experts", "num_experts_per_tok")
NUM_EXPERTS_KEYS = ("num_experts", "num_local_experts")


def find_key(d: dict, candidates: tuple[str, ...]) -> str | None:
    for k in candidates:
        if k in d:
            return k
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--top-k", type=int, default=None,
                    help="New top-k value (e.g. 6 for density-preserving on 98e)")
    ap.add_argument("--restore", action="store_true",
                    help="Restore from config.json.bak (undo)")
    args = ap.parse_args()

    d = Path(args.model_dir)
    cfg_path = d / "config.json"
    bak_path = d / "config.json.bak"
    if not cfg_path.exists():
        print(f"FAIL: {cfg_path} not found")
        return 1

    if args.restore:
        if not bak_path.exists():
            print(f"FAIL: {bak_path} not found")
            return 1
        shutil.copy(bak_path, cfg_path)
        print(f"restored {cfg_path} from {bak_path}")
        return 0

    if args.top_k is None:
        print("FAIL: --top-k required (unless --restore)")
        return 1

    with open(cfg_path) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)

    topk_key = find_key(tc, TOPK_KEYS)
    ne_key = find_key(tc, NUM_EXPERTS_KEYS)
    if topk_key is None:
        print(f"FAIL: no top-k key found (looked for {TOPK_KEYS})")
        return 2
    if ne_key is None:
        print(f"FAIL: no num-experts key found (looked for {NUM_EXPERTS_KEYS})")
        return 2

    old_topk = int(tc[topk_key])
    n_experts = int(tc[ne_key])
    if args.top_k > n_experts:
        print(f"FAIL: --top-k {args.top_k} > num_experts {n_experts}")
        return 1
    if args.top_k < 1:
        print(f"FAIL: --top-k {args.top_k} < 1")
        return 1

    # Backup
    if not bak_path.exists():
        shutil.copy(cfg_path, bak_path)
        print(f"backup: {bak_path}")

    # Write
    tc[topk_key] = int(args.top_k)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    density_old = old_topk / n_experts
    density_new = args.top_k / n_experts
    pretrain_density = 8 / 128
    print(f"OK {cfg_path}")
    print(f"   {topk_key}: {old_topk} → {args.top_k}")
    print(f"   density:   {density_old:.4f} → {density_new:.4f} "
          f"(pretrain {pretrain_density:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
