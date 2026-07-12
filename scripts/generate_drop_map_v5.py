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


def per_class_scores(cd, classes, alpha, outlier_thresh=float("inf"),
                     outlier_mode="keep", score_mode="legacy"):
    """[C, L, E] base score = wnorm × alpha + tc, indexed by expert id.

    Outlier handling (for calibration JSONs whose deep layers carry
    pathological bf16 artifacts — see project_calibration_deep_layer_corruption):
    a cell is "bad" if its wnorm is non-finite OR |wnorm| > ``outlier_thresh``.
    Because normalize="rank" uses argsort (which sorts NaN/+inf/huge to the TOP
    → rank 1.0 → protected/kept), an untreated outlier expert is silently
    *kept*. ``outlier_mode`` decides what a bad cell's score becomes BEFORE
    normalization, per (class, layer) row:
      keep   : leave as-is (legacy; outlier → top rank → kept)
      median : impute to the median of the clean cells in that row (neutral
               rank — the floor/baseline/other-class signal decides the expert)
      drop   : set to -inf (bottom rank → becomes a drop candidate)
      zero   : set to 0.0 (low rank when other scores are positive)
    """
    L = int(cd["metadata"]["num_layers"])
    E = int(cd["metadata"]["num_experts"])
    s = np.zeros((len(classes), L, E), dtype=np.float64)
    wn = np.zeros((len(classes), L, E), dtype=np.float64)
    for ci, cls in enumerate(classes):
        per_layer = cd["categories"][cls]
        for li in range(L):
            row = per_layer[str(li)]
            assert len(row) == E, f"class={cls} layer={li}: {len(row)} != {E}"
            for e in row:
                eid = int(e["id"])
                w = float(e["wnorm"])
                wn[ci, li, eid] = w
                s[ci, li, eid] = (w * alpha if score_mode == "mean"
                                  else w * alpha + float(e["tc"]))
    if outlier_mode == "keep" and np.isinf(outlier_thresh):
        return s
    bad = ~np.isfinite(s) | ~np.isfinite(wn) | (np.abs(wn) > outlier_thresh)
    n_bad = int(bad.sum())
    if n_bad:
        print(f"[v5-dropmap] outlier handling: mode={outlier_mode} "
              f"thresh={outlier_thresh:g} → {n_bad} bad cells "
              f"({100*n_bad/s.size:.1f}% of C×L×E)")
        for ci in range(len(classes)):
            for li in range(L):
                m = bad[ci, li]
                if not m.any():
                    continue
                if outlier_mode == "keep":
                    continue
                if outlier_mode == "drop":
                    s[ci, li, m] = -np.inf
                elif outlier_mode == "zero":
                    s[ci, li, m] = 0.0
                elif outlier_mode == "median":
                    clean = s[ci, li, ~m]
                    s[ci, li, m] = float(np.median(clean)) if clean.size else 0.0
                else:
                    raise ValueError(f"unknown outlier_mode: {outlier_mode}")
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


def v4_pooled_score(v4_data, alpha, outlier_thresh=float("inf"),
                    outlier_mode="keep", score_mode="legacy"):
    """v4 pooled-max-of-classes score per (L, E).

    Mirrors the v4 recipe: per-class score = wnorm*alpha + tc; take max
    across all v4 classes (math/logic/code/science/creative/...). Returns a
    [L, E] float64 array of v4-pooled scores. Drives the v4-floor protection
    (--v4-floor-top seed AND, via --v4-floor-map clamp, the 15-25/layer floor).

    Outlier handling (T176): the SAME bad-cell treatment as ``per_class_scores``
    is applied per (class, layer) row BEFORE the cross-class max-pool. This is
    the LOAD-BEARING floor path — before this guard the A2 recipe's
    ``--outlier-wnorm-thresh/--outlier-mode`` reached only the *ranking* path
    (``per_class_scores``), NOT the floor it actually protects on, so a single
    calibration outlier (non-finite or |wnorm| > thresh) could dominate the
    pooled-max and force-protect the wrong expert (or, with the legacy fp16
    base map, leave multilingual experts unprotected). ``keep`` + inf thresh
    reproduces the old raw behaviour exactly.
    """
    classes = list(v4_data["categories"].keys())
    L = int(v4_data["metadata"]["num_layers"])
    E = int(v4_data["metadata"]["num_experts"])
    C = len(classes)
    s = np.full((C, L, E), -np.inf, dtype=np.float64)
    wn = np.zeros((C, L, E), dtype=np.float64)
    for ci, cls in enumerate(classes):
        per_layer = v4_data["categories"][cls]
        for li in range(L):
            row = per_layer[str(li)]
            assert len(row) == E
            for d in row:
                eid = int(d["id"])
                w = float(d["wnorm"])
                wn[ci, li, eid] = w
                s[ci, li, eid] = (w * alpha if score_mode == "mean"
                                  else w * alpha + float(d["tc"]))
    if not (outlier_mode == "keep" and np.isinf(outlier_thresh)):
        bad = ~np.isfinite(s) | ~np.isfinite(wn) | (np.abs(wn) > outlier_thresh)
        n_bad = int(bad.sum())
        if n_bad:
            print(f"[v5-dropmap] v4-floor outlier handling: mode={outlier_mode} "
                  f"thresh={outlier_thresh:g} → {n_bad} bad cells "
                  f"({100*n_bad/s.size:.1f}% of C×L×E)")
            for ci in range(C):
                for li in range(L):
                    m = bad[ci, li]
                    if not m.any():
                        continue
                    if outlier_mode == "drop":
                        s[ci, li, m] = -np.inf
                    elif outlier_mode == "zero":
                        s[ci, li, m] = 0.0
                    elif outlier_mode == "median":
                        clean = s[ci, li, ~m]
                        s[ci, li, m] = float(np.median(clean)) if clean.size else 0.0
                    elif outlier_mode != "keep":
                        raise ValueError(f"unknown outlier_mode: {outlier_mode}")
    return s.max(axis=0)


def make_drop_map(score, target, protect_top, s_norm=None, class_protect_floor=0,
                  protect_score=None, v4_pooled=None, v4_floor_top=0,
                  v4_floor_per_layer=None):
    """Build drop map.

    `score` drives drop ranking (sorted ascending = drop-first). The
    protect-top set is computed from `protect_score` if provided (decoupled
    two-stage: e.g. sum-eq for shielding, max+TB for drop targeting), else
    from `score` itself. When `class_protect_floor > 0` and `s_norm` is given,
    each class's top-K experts per layer are also protected. When
    `v4_floor_top > 0` and `v4_pooled` is given, the top-K experts per layer
    by v4 pooled-max-of-classes are also added to the protection set —
    this is the "v4-floor" guarantee that the v5-coder map is a strict
    super-code-set of v4's keep-set (every v4-top-K expert is preserved
    regardless of v5 class-weighted score).
    """
    L, E = score.shape
    n_drop = E - target
    drop = {}
    boundary_ties = {}
    floor_added = []
    v4_floor_added = []
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
        v4_before = len(protected)
        # Resolve effective floor for this layer: per-layer map overrides scalar
        if v4_floor_per_layer is not None and v4_pooled is not None:
            this_floor = int(v4_floor_per_layer.get(li, v4_floor_per_layer.get(str(li), v4_floor_top)))
        else:
            this_floor = v4_floor_top
        if this_floor > 0 and v4_pooled is not None:
            v4_order = np.argsort(-v4_pooled[li], kind="stable")  # high → low
            for e in v4_order[:this_floor]:
                protected.add(int(e))
        v4_floor_added.append(len(protected) - v4_before)
        # If protections overflow (|protected| > target), release the lowest-priority
        # protected experts so exactly target survive. Priority = v4_pooled score when
        # available (better proxy for "v4 would keep"), else the drop-ranking score.
        max_protected = target
        if len(protected) > max_protected:
            priority = v4_pooled[li] if v4_pooled is not None else row
            prot_list = sorted(protected, key=lambda e: -priority[e])  # highest first
            keep_protected = set(prot_list[:max_protected])
            n_released = len(protected) - len(keep_protected)
            print(f"  layer {li}: |protected|={len(protected)} > target={max_protected}; "
                  f"released {n_released} lowest-priority protections")
            protected = keep_protected
        candidates = [int(e) for e in order_drop if int(e) not in protected]
        dropped = candidates[:n_drop]
        drop[li] = sorted(dropped)
        if n_drop > 0:
            boundary_score = row[dropped[-1]]
            kept_idx = [int(e) for e in order_drop[::-1] if int(e) not in set(dropped)]
            close_to_boundary = sum(1 for e in kept_idx if row[e] < boundary_score * 1.01)
            boundary_ties[li] = int(close_to_boundary)
    return drop, boundary_ties, floor_added, v4_floor_added


def load_baseline_drop_map(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return {int(k): sorted(int(x) for x in v) for k, v in d.items()}


def v4_best_class_per_layer(v4_data, alpha, score_mode="legacy"):
    """Return [L, E] np.array of strings — v4's best-class per expert per layer.

    For each (layer, expert), best-class is argmax over v4 classes of
    (wnorm * alpha + tc). Classes: math/logic/code/science/creative.
    """
    classes = list(v4_data["categories"].keys())
    L = int(v4_data["metadata"]["num_layers"])
    E = int(v4_data["metadata"]["num_experts"])
    scores = np.full((len(classes), L, E), -np.inf, dtype=np.float64)
    for ci, c in enumerate(classes):
        per_layer = v4_data["categories"][c]
        for li_s, row in per_layer.items():
            li = int(li_s)
            for d in row:
                scores[ci, li, int(d["id"])] = (
                    float(d["wnorm"]) * alpha if score_mode == "mean"
                    else float(d["wnorm"]) * alpha + float(d["tc"]))
    best_idx = scores.argmax(axis=0)  # [L, E]
    best_class = np.empty((L, E), dtype=object)
    for ci, c in enumerate(classes):
        best_class[best_idx == ci] = c
    return best_class


def enforce_class_minima(drop_map, agg, v4_best_class, min_keep_per_class,
                         protected_pinned, target):
    """In-place enforce per-v4-class minima on kept set per layer.

    For each layer, ensure at least min_keep[class] experts whose
    v4-best-class is `class` are in the kept set. If short, promote the
    highest-aggregate-score dropped experts of that class and demote the
    lowest-aggregate-score kept experts that (a) are not in protected_pinned
    and (b) are not of any class that's already at-or-below its own minimum.

    `protected_pinned` is a dict {layer: set(eid)} of experts that MUST stay
    kept (typically protect_top + v4_floor experts).
    Returns:  (drop_map_updated, swap_log)
    """
    L, E = agg.shape
    ALL = set(range(E))
    swap_log = {}
    for li in range(L):
        kept = ALL - set(drop_map[li])
        pinned = protected_pinned.get(li, set())
        # Count kept per v4-class for this layer
        kept_by_class = {}
        for e in kept:
            c = v4_best_class[li, e]
            kept_by_class.setdefault(c, set()).add(e)
        n_swaps = 0
        # For each class needing more, try to satisfy
        for cls, min_n in min_keep_per_class.items():
            cur = len(kept_by_class.get(cls, set()))
            if cur >= min_n:
                continue
            shortfall = min_n - cur
            # Promote candidates: dropped experts of this class, highest agg first
            dropped_of_cls = sorted(
                [e for e in drop_map[li] if v4_best_class[li, e] == cls],
                key=lambda e: -agg[li, e]
            )
            if len(dropped_of_cls) < shortfall:
                shortfall = len(dropped_of_cls)
            promote = dropped_of_cls[:shortfall]
            # Demote candidates: kept experts NOT in pinned, NOT of any class
            # already at its own minimum (so we don't break other class minima)
            def at_min(c):
                return len(kept_by_class.get(c, set())) <= min_keep_per_class.get(c, 0)
            demote_pool = [
                e for e in kept
                if e not in pinned
                and v4_best_class[li, e] != cls
                and not at_min(v4_best_class[li, e])
            ]
            demote_pool.sort(key=lambda e: agg[li, e])  # lowest score first
            demote = demote_pool[:shortfall]
            if len(demote) < shortfall:
                # Relax: allow demoting from classes currently AT their min if
                # we have no other choice. Worst case: skip remaining shortfall.
                extra = [
                    e for e in kept
                    if e not in pinned
                    and v4_best_class[li, e] != cls
                    and e not in set(demote)
                ]
                extra.sort(key=lambda e: agg[li, e])
                demote += extra[:shortfall - len(demote)]
            # Apply swaps
            for p, d in zip(promote, demote):
                drop_map[li].remove(p)
                drop_map[li].append(d)
                kept.discard(d)
                kept.add(p)
                kept_by_class.setdefault(cls, set()).add(p)
                dc = v4_best_class[li, d]
                if dc in kept_by_class:
                    kept_by_class[dc].discard(d)
                n_swaps += 1
            drop_map[li].sort()
            assert len(drop_map[li]) == E - target, (
                f"layer {li} class={cls} drop count drift: {len(drop_map[li])}")
        swap_log[li] = n_swaps
    return drop_map, swap_log


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
    ap.add_argument("--score-mode", choices=["legacy", "mean"], default="legacy",
                    help="T176: 'legacy' = wnorm×alpha + tc (frequency double-counted "
                         "when wnorm is a SUM). 'mean' = wnorm×alpha only — use with "
                         "RMS-wnorm maps (metadata.wnorm_mode='rms_per_token') so the "
                         "score is pure intensive competence with no routing-frequency "
                         "bias. Default legacy preserves bit-identical A2 reproduce.")
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
    ap.add_argument("--v4-floor-data", default="scripts/expert_neuron_v4.json",
                    help="v4 scoring JSON (math/logic/code/science/creative). Used "
                         "with --v4-floor-top to protect experts that v4 valued "
                         "highly when pooled across classes. Default v4 path.")
    ap.add_argument("--v4-floor-top", type=int, default=0,
                    help="Top-K experts per layer (by v4 pooled-max-of-classes "
                         "score) are added to the protection set, regardless of "
                         "v5 class-weighted score. This makes the resulting drop "
                         "map a strict super-code-set of v4's top-K keep-set — "
                         "v5-coder candidates that retain v4's breadth backbone. "
                         "Typical: 80 (= top-48 of 128). 0 disables (default). "
                         "Overridden per-layer by --v4-floor-map if both given.")
    ap.add_argument("--v4-floor-clamp", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"),
                    help="Rescale the per-layer v4-floor into [LO,HI] at consumption "
                         "time, overriding the range baked into the floor map. Uses "
                         "the map's top98_mean_per_layer signal if present (preserves "
                         "the per-layer shape past saturation), else floor_per_layer. "
                         "Use to make the v4-floor BIND at low keep-N, e.g. "
                         "`--v4-floor-clamp 40 60` for --target<=62.")
    ap.add_argument("--v4-floor-map", default=None,
                    help="Optional JSON file with per-layer v4-floor values, e.g. "
                         "`{\"floor_per_layer\": {\"0\": 95, \"1\": 85, ...}}` or "
                         "a flat `{\"0\": 95, ...}` map. When given, replaces "
                         "--v4-floor-top on a per-layer basis. Layers absent fall "
                         "back to --v4-floor-top. Use this to weight v4-floor by "
                         "per-layer relevance (e.g. scaled by top98_mean(v4)).")
    ap.add_argument("--breadth-bonus", type=float, default=0.0,
                    help="After class-weighted aggregation, add λ × mean(rank_norm) "
                         "across classes to each (L, E) cell. Rewards multi-class "
                         "generalists that don't peak on any one class but score "
                         "moderately on many — exactly the experts v4's pooled-max "
                         "kept that C1's class-weighted aggregation demoted. Try "
                         "0.3-0.5 (small bonus) up to 1.0 (equal weight to breadth "
                         "vs specialist signal). 0.0 disables (default).")
    ap.add_argument("--protect-class-min", nargs="+", default=None,
                    help="Per-v4-best-class minimum kept experts per layer, e.g. "
                         "`--protect-class-min logic:18 creative:22`. After the "
                         "v5-driven drop selection, swap in the highest-scored "
                         "dropped experts whose v4-best-class matches, until each "
                         "class meets its minimum. Demote pool: lowest-scored kept "
                         "experts not in protect_top/v4-floor and not in a class "
                         "already at its own minimum. Useful to preserve stop-signal "
                         "carriers (creative/logic generalists) that the v5 code-"
                         "weighted scoring would otherwise systematically drop.")
    ap.add_argument("--outlier-wnorm-thresh", type=float, default=float("inf"),
                    help="Treat a cell as a calibration outlier if |wnorm| > this "
                         "(non-finite always counts). Default inf = off. Use 1e6 "
                         "to filter the v3 deep-layer bf16 artifacts (3.9e18). See "
                         "project_calibration_deep_layer_corruption.")
    ap.add_argument("--outlier-mode",
                    choices=["keep", "median", "drop", "zero"], default="keep",
                    help="What an outlier cell's score becomes before rank-normalize: "
                         "keep=legacy (argsort floats it to top→kept); median=neutral "
                         "rank; drop=-inf (drop candidate); zero. Default keep.")
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

    s = per_class_scores(cd, classes, args.alpha,
                         outlier_thresh=args.outlier_wnorm_thresh,
                         outlier_mode=args.outlier_mode, score_mode=args.score_mode)
    s_norm = normalize_per_class_per_layer(s, mode=args.normalize)
    print(f"[v5-dropmap] normalize={args.normalize}")
    agg = aggregate(s_norm, weights, args.strategy)

    # Breadth bonus: add λ · mean(rank_norm across classes) to each (L, E).
    # Rewards experts that score moderately on many classes (multi-class
    # generalists v4 keeps but C1 demotes).
    if args.breadth_bonus > 0:
        breadth = s_norm.mean(axis=0)  # [L, E]
        agg = agg + float(args.breadth_bonus) * breadth
        print(f"[v5-dropmap] breadth_bonus={args.breadth_bonus} "
              f"(breadth term added to aggregate; mean over classes of rank_norm)")

    # Load v4 floor scoring if requested
    v4_pooled = None
    v4_floor_per_layer = None
    need_v4_pooled = args.v4_floor_top > 0 or args.v4_floor_map is not None
    if need_v4_pooled:
        v4_path = args.v4_floor_data
        print(f"[v5-dropmap] loading v4 scoring from {v4_path} for v4-floor protection ...")
        with open(v4_path) as f:
            v4_data = json.load(f)
        v4_pooled = v4_pooled_score(v4_data, args.alpha,
                                    outlier_thresh=args.outlier_wnorm_thresh,
                                    outlier_mode=args.outlier_mode,
                                    score_mode=args.score_mode)
    if args.v4_floor_map is not None:
        with open(args.v4_floor_map) as f:
            fm_raw = json.load(f)
        # Accept either nested ("floor_per_layer": {...}) or flat ({"0": 95, ...})
        if isinstance(fm_raw, dict) and "floor_per_layer" in fm_raw:
            v4_floor_per_layer = {int(k): int(v) for k, v in fm_raw["floor_per_layer"].items()}
        else:
            v4_floor_per_layer = {int(k): int(v) for k, v in fm_raw.items()}
        if args.v4_floor_clamp is not None:
            lo, hi = args.v4_floor_clamp
            src = fm_raw.get("top98_mean_per_layer") if isinstance(fm_raw, dict) else None
            raw = ({int(k): float(v) for k, v in src.items()} if src
                   else {k: float(v) for k, v in v4_floor_per_layer.items()})
            rmin, rmax = min(raw.values()), max(raw.values())
            span = (rmax - rmin) or 1.0
            v4_floor_per_layer = {
                k: max(lo, min(hi, int(round(lo + (hi - lo) * (v - rmin) / span))))
                for k, v in raw.items()}
            print(f"[v5-dropmap] --v4-floor-clamp {lo} {hi} → rescaled per-layer floor "
                  f"(source={'top98_mean' if src else 'floor_per_layer'})")
        vals = list(v4_floor_per_layer.values())
        print(f"[v5-dropmap] v4_floor_map={args.v4_floor_map} → "
              f"per-layer floor min={min(vals)} max={max(vals)} mean={sum(vals)/len(vals):.1f}")
    elif args.v4_floor_top > 0:
        print(f"[v5-dropmap] v4_floor_top={args.v4_floor_top} → "
              f"top-K v4-pooled experts per layer are protected")

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
    drop, boundary_ties, floor_added, v4_floor_added = make_drop_map(
        agg, args.target, args.protect_top,
        s_norm=s_norm if args.class_protect_floor > 0 else None,
        class_protect_floor=args.class_protect_floor,
        protect_score=protect_agg,
        v4_pooled=v4_pooled,
        v4_floor_top=args.v4_floor_top,
        v4_floor_per_layer=v4_floor_per_layer,
    )
    if args.class_protect_floor > 0:
        print(f"[v5-dropmap] class_protect_floor={args.class_protect_floor} → "
              f"extra-protected per layer: mean={np.mean(floor_added):.1f} "
              f"min={min(floor_added)} max={max(floor_added)}")
    if args.v4_floor_top > 0:
        print(f"[v5-dropmap] v4_floor_top={args.v4_floor_top} → "
              f"extra-protected per layer by v4-pooled rank: "
              f"mean={np.mean(v4_floor_added):.1f} "
              f"min={min(v4_floor_added)} max={max(v4_floor_added)}")

    # --- v4-class-min enforcement (post-process drop map) ---
    if args.protect_class_min:
        # Parse "class:N" pairs
        min_keep = {}
        for tok in args.protect_class_min:
            k, _, v = tok.partition(":")
            min_keep[k.strip()] = int(v)
        if v4_pooled is None:
            # Need v4_data loaded; load now if not already
            with open(args.v4_floor_data) as f:
                v4_data_for_class = json.load(f)
        else:
            with open(args.v4_floor_data) as f:
                v4_data_for_class = json.load(f)
        valid_classes = set(v4_data_for_class["categories"].keys())
        unknown = [c for c in min_keep if c not in valid_classes]
        if unknown:
            raise SystemExit(
                f"--protect-class-min: unknown v4 classes {unknown}; "
                f"valid: {sorted(valid_classes)}")
        v4_bc = v4_best_class_per_layer(v4_data_for_class, args.alpha,
                                        score_mode=args.score_mode)
        # Build per-layer pinned set: protect_top (by aggregate) + v4_floor experts
        pinned = {}
        for li in range(L):
            row = agg[li]
            ps = (protect_agg[li] if protect_agg is not None else row)
            top_prot = set(int(e) for e in np.argsort(-ps, kind="stable")[:args.protect_top])
            if v4_pooled is not None:
                this_floor = args.v4_floor_top
                if v4_floor_per_layer is not None:
                    this_floor = int(v4_floor_per_layer.get(li, v4_floor_per_layer.get(str(li), args.v4_floor_top)))
                if this_floor > 0:
                    v4_order = np.argsort(-v4_pooled[li], kind="stable")
                    for e in v4_order[:this_floor]:
                        top_prot.add(int(e))
            pinned[li] = top_prot
        # Convert drop_map from set-like to list for in-place modification
        drop_lists = {li: list(drop[li]) for li in range(L)}
        drop_lists, swap_log = enforce_class_minima(
            drop_lists, agg, v4_bc, min_keep, pinned, args.target)
        drop = drop_lists
        total_swaps = sum(swap_log.values())
        print(f"[v5-dropmap] protect_class_min={min_keep} → "
              f"applied {total_swaps} swaps total "
              f"({total_swaps/L:.1f}/layer; max layer={max(swap_log.values())})")
        # Re-validate per-class counts
        kept_class_counts_post = {c: 0 for c in min_keep}
        ALL = set(range(E))
        for li in range(L):
            kept = ALL - set(drop[li])
            for e in kept:
                c = v4_bc[li, e]
                if c in kept_class_counts_post:
                    kept_class_counts_post[c] += 1
        print(f"[v5-dropmap] post-enforcement kept totals: "
              f"{ {c: kept_class_counts_post[c] for c in min_keep} } "
              f"(min×30 layers = { {c: 30*min_keep[c] for c in min_keep} })")

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
