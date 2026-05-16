#!/usr/bin/env python3
"""generate_drop_map_v5.py — produce a per-layer expert drop map for v5 mapping data.

Reads `scripts/expert_neuron_v5_code.json` (or sister variants). Categories
include 5 generic_* (math/logic/code/science/creative) and 3 targeted_*
(humaneval / humanevalplus / lcb_medium_55). The targeted_* categories are
the code-specific PASS-trace replays that *define* the v5-code variant —
they are weighted higher by default (matches `tier_b_set_weights.B = 3.0`).

Score is per-class wnorm × alpha + tc, normalized per layer per class to
equalize scales across classes (some classes have larger absolute wnorm),
then aggregated via {max, mean, geomean, sum} with optional per-class weights.

Drop policy: per layer, drop the (num_experts - target) lowest-scoring experts,
shielding the top-N highest-scoring (protect-top) from being dropped.

Output JSON shape: {layer_str: [list-of-dropped-expert-ids]} — drop-in
compatible with `scripts/expert_drop.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Default per-class weights for v5 data. Generic categories contribute the
# breadth signal; targeted_* categories contribute the code-specialist signal
# (already weighted 3× during Tier-B collection — we mirror that here).
DEFAULT_WEIGHTS_V5 = {
    "generic_math":          1.0,
    "generic_logic":         1.0,
    "generic_code":          1.0,
    "generic_science":       1.0,
    "generic_creative":      1.0,
    "targeted_humaneval":    3.0,
    "targeted_humanevalplus":3.0,
    "targeted_lcb_medium_55":3.0,
}


def per_class_scores(cd, classes, alpha):
    """[C, L, E] base score = wnorm × alpha + tc, indexed by expert id."""
    L = int(cd["metadata"]["num_layers"])
    E = int(cd["metadata"]["num_experts"])
    s = np.zeros((len(classes), L, E), dtype=np.float64)
    for ci, cls in enumerate(classes):
        per_layer = cd["categories"][cls]
        for li in range(L):
            row = per_layer[str(li)]
            assert len(row) == E, f"class={cls} layer={li}: {len(row)} != {E}"
            for e in row:
                s[ci, li, int(e["id"])] = float(e["wnorm"]) * alpha + float(e["tc"])
    return s


def normalize_per_class_per_layer(s, mode="rank", eps=1e-12):
    """Make each class contribute on a common scale per layer.

    mode="rank" (DEFAULT — heavy-tail safe): replace each [c,l,:] slice with
        percentile-ranks in [0, 1]. Distribution-shape invariant; explicit
        tier weights actually dominate the max as intended.
    mode="mean": legacy v4 behavior — divide by per-class per-layer mean. Fails
        when one class has a heavy tail (e.g. generic_math with a few huge
        outliers pulls the mean up, suppressing the rest of the class).
    """
    if mode == "mean":
        means = s.mean(axis=2, keepdims=True)
        return s / np.clip(means, eps, None)
    if mode == "rank":
        # argsort twice = ranks; divide by (E-1) → [0, 1]
        C, L, E = s.shape
        out = np.empty_like(s)
        for ci in range(C):
            for li in range(L):
                order = np.argsort(s[ci, li], kind="stable")
                ranks = np.empty(E, dtype=np.float64)
                ranks[order] = np.arange(E, dtype=np.float64)
                out[ci, li] = ranks / max(E - 1, 1)
        return out
    raise ValueError(f"unknown normalize mode: {mode}")


def aggregate(s_norm, weights, strategy):
    """Aggregate per-class scores [C, L, E] → [L, E] using per-class weights.

    Strategies in order of specialist-rescue strength (rank+max+uniform is
    the most lenient, mean/geomean the least). The choice fundamentally
    decides which experts get rescued by a single-class win:

      max        : save if ANY 1 class ranks expert high (most lenient)
      lp4        : (Σ w·s⁴)^¼ — smooth approx of max; specialist-leaning
      softmax_t4 : (1/τ)·log Σ exp(τ·w·s), τ=4 — dial between mean and max
      second     : 2nd-highest weighted class score (must be useful in ≥2)
      top3mean   : mean of weighted top-3 class scores (≥3 useful classes)
      mean / sum : balanced average (mean = sum / Σw); weakest rescue
      geomean    : weighted geomean — outlier-robust, less specialist-friendly
    """
    w = weights[:, None, None]  # broadcast to [C, 1, 1]
    if strategy == "max":
        # Weighted max: scale before max so higher-weight classes can dominate.
        return (s_norm * w).max(axis=0)
    if strategy == "mean":
        return (s_norm * w).sum(axis=0) / max(w.sum(), 1e-12)
    if strategy == "geomean":
        # Weighted geomean = exp(Σ w_c log s_c / Σ w_c)
        wsum = max(w.sum(), 1e-12)
        return np.exp((w * np.log(np.clip(s_norm, 1e-12, None))).sum(axis=0) / wsum)
    if strategy == "sum":
        return (s_norm * w).sum(axis=0)
    if strategy == "lp4":
        # (Σ w_c · s_c^4)^(1/4) — between mean (p=1) and max (p→∞).
        # Heavily weights large class scores while not collapsing to the max.
        return np.power((w * (s_norm ** 4)).sum(axis=0), 0.25)
    if strategy == "softmax_t4":
        # Boltzmann soft-max with temperature τ=4: (1/τ)·log Σ exp(τ·w_c·s_c).
        # τ→0 ≈ mean, τ→∞ ≈ max; τ=4 puts ~80% of mass on the top 2 classes
        # for typical rank-normalized inputs.
        tau = 4.0
        weighted = w * s_norm  # [C, L, E]
        m = weighted.max(axis=0, keepdims=True)  # numerical stability
        return m[0] + (1.0 / tau) * np.log(np.exp(tau * (weighted - m)).sum(axis=0))
    if strategy == "second":
        # 2nd-highest weighted class score per (l, e). Requires the expert to
        # rank high in at least TWO classes — kills single-class-specialist rescue.
        weighted = w * s_norm  # [C, L, E]
        # np.partition: kth element on axis 0; for "2nd-highest", k=-2 puts the
        # 2nd-largest at index -2 (top-2 unordered at end after partition).
        if weighted.shape[0] < 2:
            return weighted[0]
        sorted_desc = -np.sort(-weighted, axis=0)  # high→low along class axis
        return sorted_desc[1]  # 2nd row
    if strategy == "top3mean":
        # Mean of the weighted top-3 class scores per (l, e). Requires the
        # expert to rank high in at least 3 classes for high aggregate.
        weighted = w * s_norm  # [C, L, E]
        k = min(3, weighted.shape[0])
        sorted_desc = -np.sort(-weighted, axis=0)
        return sorted_desc[:k].mean(axis=0)
    raise ValueError(f"unknown strategy: {strategy}")


def make_drop_map(score, target, protect_top, s_norm=None, class_protect_floor=0,
                  protect_score=None):
    """Build drop map.

    `score` drives drop ranking (sorted ascending = drop-first). The
    protect-top set is computed from `protect_score` if provided (decoupled
    two-stage: e.g. sum-eq for shielding, max+TB for drop targeting), else
    from `score` itself. When `class_protect_floor > 0` and `s_norm` is given,
    each class's top-K experts per layer are also protected.
    """
    L, E = score.shape
    n_drop = E - target
    drop = {}
    boundary_ties = {}
    floor_added = []
    ps = protect_score if protect_score is not None else score
    for li in range(L):
        row = score[li]
        prow = ps[li]
        # Drop ranking uses `row` (low → high), protection uses `prow` (high → low)
        order_drop = np.argsort(row, kind="stable")  # low → high (drop-first)
        order_prot = np.argsort(-prow, kind="stable")  # high → low
        protected = set(int(e) for e in order_prot[:protect_top])
        before = len(protected)
        if class_protect_floor > 0 and s_norm is not None:
            for ci in range(s_norm.shape[0]):
                c_order = np.argsort(-s_norm[ci, li], kind="stable")
                for e in c_order[:class_protect_floor]:
                    protected.add(int(e))
        floor_added.append(len(protected) - before)
        candidates = [int(e) for e in order_drop if int(e) not in protected]
        dropped = candidates[:n_drop]
        drop[li] = sorted(dropped)
        if n_drop > 0:
            boundary_score = row[dropped[-1]]
            kept_idx = [int(e) for e in order_drop[::-1] if int(e) not in set(dropped)]
            close_to_boundary = sum(1 for e in kept_idx if row[e] < boundary_score * 1.01)
            boundary_ties[li] = int(close_to_boundary)
    return drop, boundary_ties, floor_added


def load_baseline_drop_map(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return {int(k): sorted(int(x) for x in v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="scripts/expert_neuron_v5_code.json",
                    help="v5 mapping JSON")
    ap.add_argument("--target", type=int, required=True,
                    help="Experts to keep per layer (e.g. 98)")
    ap.add_argument("--protect-top", type=int, default=16,
                    help="Top-N highest scorers per layer are never dropped")
    ap.add_argument("--alpha", type=float, default=2.0,
                    help="Score = wnorm × alpha + tc")
    ap.add_argument("--strategy",
                    choices=["max", "mean", "geomean", "sum",
                             "lp4", "softmax_t4", "second", "top3mean"],
                    default="max",
                    help="Aggregation across classes. See aggregate() docstring "
                         "for the specialist-rescue ladder.")
    ap.add_argument("--normalize", choices=["rank", "mean"], default="rank",
                    help="Per-class per-layer normalization. 'rank' (default) is "
                         "heavy-tail safe and lets explicit weights dominate. "
                         "'mean' is the legacy v4 unit-mean (broken when a class "
                         "has outliers — e.g. generic_math).")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="Categories to use (default: all 8 v5 categories)")
    ap.add_argument("--class-weights", nargs="+", default=None,
                    help="Per-class weights matching --classes (default: 1.0 for "
                         "generic_*, 3.0 for targeted_*)")
    ap.add_argument("--protect-strategy",
                    choices=["same", "sum", "max", "mean", "geomean",
                             "lp4", "softmax_t4", "second", "top3mean"],
                    default="same",
                    help="Aggregation for the protect-top set. 'same' uses --strategy "
                         "(default; legacy behavior). Two-stage example: --strategy max "
                         "--protect-strategy sum keeps generalists shielded while letting "
                         "max+TB drive drop ranking.")
    ap.add_argument("--protect-class-weights", nargs="+", default=None,
                    help="Per-class weights for the protect aggregation (default: all 1.0). "
                         "Used only when --protect-strategy != 'same'.")
    ap.add_argument("--class-protect-floor", type=int, default=0,
                    help="Per-class top-K experts per layer are also protected "
                         "(in addition to --protect-top by aggregate). Guarantees "
                         "every class keeps K seats; useful when one tier dominates "
                         "the aggregate and starves another class.")
    ap.add_argument("--baseline-drop-map",
                    default="scripts/teacher_force_98e_p16_clean.json",
                    help="Reference drop map for overlap stats (optional)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary-output", default=None,
                    help="Sidecar JSON with per-layer stats; default <output>.summary.json")
    args = ap.parse_args()

    with open(args.data) as f:
        cd = json.load(f)

    available_classes = cd["metadata"]["categories"]
    classes = args.classes if args.classes else list(available_classes)
    missing = [c for c in classes if c not in available_classes]
    if missing:
        raise SystemExit(f"classes not in data: {missing}; available={available_classes}")

    if args.class_weights:
        if len(args.class_weights) != len(classes):
            raise SystemExit(
                f"--class-weights len {len(args.class_weights)} != --classes len {len(classes)}")
        weights = np.array([float(w) for w in args.class_weights], dtype=np.float64)
    else:
        weights = np.array(
            [DEFAULT_WEIGHTS_V5.get(c, 1.0) for c in classes], dtype=np.float64)

    print(f"[v5-dropmap] data={args.data}")
    print(f"[v5-dropmap] classes={classes}")
    print(f"[v5-dropmap] weights={weights.tolist()}")
    print(f"[v5-dropmap] target={args.target} protect_top={args.protect_top} "
          f"alpha={args.alpha} strategy={args.strategy}")

    s = per_class_scores(cd, classes, args.alpha)
    s_norm = normalize_per_class_per_layer(s, mode=args.normalize)
    print(f"[v5-dropmap] normalize={args.normalize}")
    agg = aggregate(s_norm, weights, args.strategy)

    L, E = agg.shape
    n_drop = E - args.target
    protect_agg = None
    if args.protect_strategy != "same":
        if args.protect_class_weights:
            if len(args.protect_class_weights) != len(classes):
                raise SystemExit(
                    f"--protect-class-weights len {len(args.protect_class_weights)} "
                    f"!= --classes len {len(classes)}")
            pw = np.array([float(w) for w in args.protect_class_weights], dtype=np.float64)
        else:
            pw = np.ones(len(classes), dtype=np.float64)
        protect_agg = aggregate(s_norm, pw, args.protect_strategy)
        print(f"[v5-dropmap] protect_strategy={args.protect_strategy} "
              f"protect_weights={pw.tolist()}")
    drop, boundary_ties, floor_added = make_drop_map(
        agg, args.target, args.protect_top,
        s_norm=s_norm if args.class_protect_floor > 0 else None,
        class_protect_floor=args.class_protect_floor,
        protect_score=protect_agg,
    )
    if args.class_protect_floor > 0:
        print(f"[v5-dropmap] class_protect_floor={args.class_protect_floor} → "
              f"extra-protected per layer: mean={np.mean(floor_added):.1f} "
              f"min={min(floor_added)} max={max(floor_added)}")

    # Compare to baseline if available
    baseline = load_baseline_drop_map(args.baseline_drop_map) if args.baseline_drop_map else None
    overlap_counts = []
    per_layer_summary = {}
    for li in range(L):
        row_dropped = set(drop[li])
        row = agg[li]
        stats = {
            "n_dropped": len(row_dropped),
            "agg_min": float(row.min()),
            "agg_max": float(row.max()),
            "agg_mean": float(row.mean()),
            "boundary_ties_within_1pct": boundary_ties.get(li, 0),
        }
        if baseline is not None and li in baseline:
            base_set = set(baseline[li])
            overlap = len(row_dropped & base_set)
            stats["overlap_vs_baseline"] = overlap
            stats["non_overlap_only_in_v5"] = sorted(row_dropped - base_set)
            overlap_counts.append(overlap)
        per_layer_summary[li] = stats

    out = {str(li): drop[li] for li in range(L)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[v5-dropmap] wrote {args.output} ({L} layers × {n_drop} dropped each)")

    summary = {
        "args": vars(args),
        "classes_used": classes,
        "weights_used": weights.tolist(),
        "metadata": cd["metadata"],
        "per_layer": per_layer_summary,
    }
    if overlap_counts:
        summary["overlap_summary"] = {
            "mean_overlap_per_layer": float(np.mean(overlap_counts)),
            "min": int(min(overlap_counts)),
            "max": int(max(overlap_counts)),
            "n_drop": n_drop,
        }
        print(f"[v5-dropmap] mean overlap with {args.baseline_drop_map}: "
              f"{summary['overlap_summary']['mean_overlap_per_layer']:.2f}/{n_drop} "
              f"({100*summary['overlap_summary']['mean_overlap_per_layer']/n_drop:.1f}%)")
    sum_path = args.summary_output or (args.output + ".summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[v5-dropmap] wrote summary {sum_path}")


if __name__ == "__main__":
    main()
