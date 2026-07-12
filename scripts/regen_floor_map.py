#!/usr/bin/env python3
"""T176 — regenerate v4_layer_floor_map.json from a (rebuilt) BASE competence map.

The drop-map consumer (`generate_drop_map_v5.py`) force-protects 15-25 experts
per layer via `--v4-floor-map ... --v4-floor-clamp 15 25`. The clamp reads the
floor map's `top98_mean_per_layer` and rescales it linearly into [lo, hi] — so
only the RELATIVE per-layer ordering of top98_mean matters, not its absolute
scale. `top98_mean_per_layer[L]` = mean of the top-K (default 98 of 128)
experts' v4-pooled score at layer L: a "how strong is this layer's surviving
cohort" signal. Stronger layers → higher floor → more experts protected.

CRITICAL (T176): we import the SAME `v4_pooled_score` the consumer uses, with
the SAME `--outlier-wnorm-thresh/--outlier-mode`, so the floor map and the
runtime floor protection are computed from byte-identical scores. A floor map
built from raw (un-guarded) pooled scores while the consumer guards them (or
vice-versa) would silently desync the per-layer floor from the protection set.

Deterministic, no GPU. Output schema matches the legacy map:
`{description, floor_per_layer, top98_mean_per_layer}`.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_drop_map_v5 import v4_pooled_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-map", required=True,
                    help="rebuilt BASE competence map (expert_neuron_base_v6.json)")
    ap.add_argument("--out", default="v4_layer_floor_map_v6.json")
    ap.add_argument("--alpha", type=float, default=2.0,
                    help="wnorm weight in pooled score (MUST match the drop-map "
                         "recipe's --alpha; A2 uses 2.0).")
    ap.add_argument("--top-k", type=int, default=98,
                    help="experts per layer averaged for top98_mean (of num_experts).")
    ap.add_argument("--clamp", type=int, nargs=2, default=[15, 25],
                    metavar=("LO", "HI"),
                    help="floor_per_layer preview range (consumer re-derives this "
                         "from top98_mean at drop-map time; emitted here for audit).")
    # mirror the recipe's outlier policy so floor == protection scoring
    ap.add_argument("--outlier-wnorm-thresh", type=float, default=10000.0)
    ap.add_argument("--outlier-mode", default="median",
                    choices=["keep", "median", "drop", "zero"])
    ap.add_argument("--score-mode", default="legacy", choices=["legacy", "mean"],
                    help="T176: MUST match the drop-map recipe's --score-mode so the "
                         "floor's top98_mean ordering is computed from byte-identical "
                         "scores as the runtime v4-floor protection. 'mean' for RMS maps.")
    args = ap.parse_args()

    with open(args.base_map) as f:
        base = json.load(f)
    cats = list(base["categories"].keys())
    L = int(base["metadata"]["num_layers"])
    E = int(base["metadata"]["num_experts"])
    print(f"[floor-regen] base={args.base_map}")
    print(f"[floor-regen] categories={cats}  L={L} E={E} top_k={args.top_k} alpha={args.alpha}")
    if args.top_k > E:
        raise SystemExit(f"--top-k {args.top_k} > num_experts {E}")

    pooled = v4_pooled_score(base, args.alpha,
                             outlier_thresh=args.outlier_wnorm_thresh,
                             outlier_mode=args.outlier_mode,
                             score_mode=args.score_mode)  # [L, E]
    if not np.isfinite(pooled).all():
        n_nf = int((~np.isfinite(pooled)).sum())
        print(f"[floor-regen] WARNING: {n_nf} non-finite pooled cells "
              f"(expected only if outlier_mode=drop zeroed a whole expert across all classes)")

    top98 = {}
    for li in range(L):
        row = np.sort(pooled[li])[::-1]          # high → low
        topk = row[:args.top_k]
        topk = topk[np.isfinite(topk)]           # ignore -inf (drop'd) cells in the mean
        top98[li] = float(topk.mean()) if topk.size else 0.0

    # preview floor_per_layer with the same linear rescale the consumer applies
    lo, hi = args.clamp
    vmin, vmax = min(top98.values()), max(top98.values())
    span = (vmax - vmin) or 1.0
    floor = {li: int(round(lo + (hi - lo) * (top98[li] - vmin) / span)) for li in range(L)}

    out = {
        "description": f"T176 v4-floor from {Path(args.base_map).name}: top{args.top_k}_mean "
                       f"of guarded v4-pooled score (alpha={args.alpha}, "
                       f"outlier={args.outlier_mode}@{args.outlier_wnorm_thresh:g}); "
                       f"floor_per_layer preview rescaled to [{lo},{hi}].",
        "floor_per_layer": {str(li): floor[li] for li in range(L)},
        "top98_mean_per_layer": {str(li): top98[li] for li in range(L)},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    ranked = sorted(range(L), key=lambda li: top98[li], reverse=True)
    print(f"[floor-regen] wrote {args.out}")
    print(f"  top98_mean range: min={vmin:.2f} (L{min(top98, key=top98.get)}) "
          f"max={vmax:.2f} (L{max(top98, key=top98.get)})")
    print(f"  strongest layers (high floor): {ranked[:5]}")
    print(f"  weakest   layers (low  floor): {ranked[-5:]}")
    print(f"  floor_per_layer preview min={min(floor.values())} max={max(floor.values())}")


if __name__ == "__main__":
    main()
