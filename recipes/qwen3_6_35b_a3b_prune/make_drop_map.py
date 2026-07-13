#!/usr/bin/env python3
"""
Derive an expert-drop map from a competence map (the gate_competence_map.py format)
for Qwen3.6-35B-A3B. Ranks experts per layer by an importance score aggregated across
categories, drops the lowest `--drop-count` per layer, and writes the drop-map JSON that
expert_drop_qwen35b.py consumes: {"0":[ids],...,"39":[ids],"mtp":[ids]}.

Competence-map input (produced by the profiler, validated by gate_competence_map.py):
  {"metadata":{"num_layers":L,"num_experts":E},
   "categories":{<cat>:{"<li>":[{"id":e,"wnorm":..,"tc":..,"neuron_act":[..]?}, ...E], ...L}, ...}}

Importance score per (layer, expert), aggregated over categories with --agg:
  tc         routing frequency (times selected in top-k over the calib corpus)  [default, REAP-style]
  wnorm      per-expert output-contribution norm
  wnorm_tc   wnorm * tc  (magnitude x frequency)
Lowest-importance experts per layer are dropped (least-used / least-contributing).

Cross-category aggregation --agg:
  sum | max | mean   RAW aggregation over categories (weights ignored). Domain-BLIND
                     with sum (== global frequency, the REAP baseline used by the 184e).
  wmax | wsum        WEIGHTED aggregation for TARGETED (coder/specialist) variants. Each
                     category's per-layer score is rank-normalized to [0,1] (so weights are
                     comparable across categories with different token counts), then combined:
                       wmax : score = max_cat( weight[cat] * ranknorm[cat] )   (v7-coder default)
                       wsum : score = sum_cat( weight[cat]*ranknorm[cat] ) / sum(weight)
                     Per-category weights come from --cat-weight (default 1.0). This mirrors the
                     Gemma v7-coder T17 "targeted" strategy: with w_targeted >> w_generic a
                     target-domain PASS specialist is guaranteed to outrank a generic-only expert
                     at the drop boundary, but generic competence is only outranked, never dropped.

Targeting floor (--floor-count F): guarantee that the F most base-critical experts per layer
  SURVIVE the drop even if targeting would evict them (the v7 floor-clamp guarantee, adapted to
  this uniform per-layer drop). Base criticality is the untargeted channel: the sum of
  rank-normalized scores over --floor-cats (default = every category whose weight is 1.0, i.e.
  the generic categories) taken from --floor-map (default = the competence map itself). The top-F
  by that base score are force-kept; the drop-count is taken from the remaining, targeting-ranked
  experts. Requires F <= E - drop-count. F=0 (default) disables the floor.
  Note: v7's `--v4-floor-clamp LO HI` was a per-layer KEEP-count band for the Gemma global-drop
  mode; this dropper drops a fixed count per layer, so the equivalent is a single --floor-count
  (use e.g. the clamp's HI as the guaranteed floor, then tune empirically).

MTP head (--mtp-strategy):
  global  drop the bottom-`drop-count` by importance aggregated over ALL main layers  [default]
  layer0  reuse layer 0's drop set
  none    omit "mtp"  (expert_drop_qwen35b.py then defaults MTP to layer 0's set)

Usage:
  # balanced REAP baseline (reproduces drop_map_184e.json byte-identically):
  python make_drop_map.py --competence-map results/competence_qwen35b.json \
      --drop-count 72 --score tc --agg sum --mtp-strategy global \
      -o results/drop_map_184e.json

  # LCB-targeted coder variant (v7-coder-style: targeted_lcb weighted, generic floor):
  python make_drop_map.py --competence-map results/competence_qwen35b_lcb.json \
      --drop-count 72 --score tc --agg wmax \
      --cat-weight targeted_lcb=2.0 --floor-count 40 \
      --mtp-strategy global -o results/drop_map_184e_lcb.json
"""
import argparse
import json
from collections import defaultdict

RAW_AGGS = ("sum", "max", "mean")
WEIGHTED_AGGS = ("wmax", "wsum")


def score_of(expert, kind):
    if kind == "tc":
        return float(expert.get("tc", 0.0))
    if kind == "wnorm":
        return float(expert.get("wnorm", 0.0))
    if kind == "wnorm_tc":
        return float(expert.get("wnorm", 0.0)) * float(expert.get("tc", 0.0))
    raise ValueError(f"unknown score {kind}")


def aggregate(values, how):
    if not values:
        return 0.0
    if how == "sum":
        return sum(values)
    if how == "max":
        return max(values)
    if how == "mean":
        return sum(values) / len(values)
    raise ValueError(f"unknown agg {how}")


def parse_cat_weights(items):
    """--cat-weight CAT=W (repeatable) -> {cat: float}. Unlisted categories default to 1.0."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--cat-weight expects CAT=W, got: {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            raise SystemExit(f"--cat-weight {it!r}: {v!r} is not a float")
    return out


def rank_normalize(cat_scores, E):
    """cat_scores: {eid: raw_score} for one (category, layer). Returns {eid: rank01} in [0,1],
    where the highest-scoring expert -> 1.0 and the lowest -> 0.0 (ordinal rank / (E-1)).
    Missing experts score 0.0 before ranking. Deterministic tie-break on eid."""
    ordered = sorted(range(E), key=lambda e: (cat_scores.get(e, 0.0), e))  # ascending
    denom = max(E - 1, 1)
    return {e: idx / denom for idx, e in enumerate(ordered)}


def load_map(path):
    with open(path) as f:
        cm = json.load(f)
    return cm


def raw_per_cat_layer(cm, cats, L, E, score):
    """-> {cat: {li: {eid: raw_score}}} for the requested categories."""
    out = {}
    for cat in cats:
        layers = cm["categories"][cat]
        cat_out = {}
        for li in range(L):
            row = layers[str(li)]
            assert len(row) == E, f"{cat} L{li}: {len(row)} experts != {E}"
            cat_out[li] = {int(e["id"]): score_of(e, score) for e in row}
        out[cat] = cat_out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--competence-map", required=True)
    ap.add_argument("--drop-count", type=int, default=72)
    ap.add_argument("--score", choices=["tc", "wnorm", "wnorm_tc"], default="tc")
    ap.add_argument("--agg", choices=RAW_AGGS + WEIGHTED_AGGS, default="sum")
    ap.add_argument("--cat-weight", action="append", default=[], metavar="CAT=W",
                    help="Per-category weight for wmax/wsum (repeatable). Unlisted -> 1.0.")
    ap.add_argument("--floor-count", type=int, default=0,
                    help="Force-keep the top-F base-critical experts/layer (v7 floor guarantee). 0=off.")
    ap.add_argument("--floor-map", default=None,
                    help="Competence map used for the base/floor ranking (default: --competence-map).")
    ap.add_argument("--floor-cats", default=None,
                    help="Comma-separated categories forming the untargeted base channel for the floor "
                         "(default: every category whose --cat-weight is 1.0).")
    ap.add_argument("--mtp-strategy", choices=["global", "layer0", "none"], default="global")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    weights = parse_cat_weights(args.cat_weight)
    weighted = args.agg in WEIGHTED_AGGS
    if weights and not weighted:
        raise SystemExit(f"--cat-weight is only meaningful with --agg wmax|wsum (got --agg {args.agg}); "
                         f"raw aggs {RAW_AGGS} ignore weights.")

    cm = load_map(args.competence_map)
    L = int(cm["metadata"]["num_layers"])
    E = int(cm["metadata"]["num_experts"])
    cats = list(cm["categories"].keys())
    # Guard against the silent no-op trap: the Tier-C corpus producer names categories
    # `corpus_<bench>` (e.g. corpus_targeted_lcb), so a bare `--cat-weight targeted_lcb=2.0`
    # would match nothing and weight everything 1.0. Fail loud with the available names.
    unknown = set(weights) - set(cats)
    if unknown:
        raise SystemExit(f"--cat-weight names categories not in the map: {sorted(unknown)}\n"
                         f"available categories: {cats}")
    if not (0 < args.drop_count < E):
        raise SystemExit(f"drop-count {args.drop_count} must be in (0,{E})")
    keep = E - args.drop_count
    if not (0 <= args.floor_count <= keep):
        raise SystemExit(f"--floor-count {args.floor_count} must be in [0,{keep}] (E-drop_count)")

    w = {c: float(weights.get(c, 1.0)) for c in cats}
    raw = raw_per_cat_layer(cm, cats, L, E, args.score)

    # ── importance[li][eid] : targeting-aware per-layer importance ────────────────
    importance = defaultdict(dict)   # li -> {eid: score}
    global_imp = defaultdict(list)   # eid -> [per-layer scores]  (for MTP global)
    if weighted:
        # rank-normalize each (cat, layer) to [0,1], then weighted-combine across cats.
        norm = {c: {li: rank_normalize(raw[c][li], E) for li in range(L)} for c in cats}
        wsum_all = sum(w[c] for c in cats) or 1e-12
        for li in range(L):
            for e in range(E):
                if args.agg == "wmax":
                    s = max(w[c] * norm[c][li][e] for c in cats)
                else:  # wsum
                    s = sum(w[c] * norm[c][li][e] for c in cats) / wsum_all
                importance[li][e] = s
                global_imp[e].append(s)
    else:
        # RAW aggregation across categories (back-compat; sum == REAP baseline).
        for li in range(L):
            for e in range(E):
                vals = [raw[c][li][e] for c in cats]
                s = aggregate(vals, args.agg)
                importance[li][e] = s
                global_imp[e].append(s)

    # ── floor: force-keep the top-F base-critical experts per layer ───────────────
    protected = {li: set() for li in range(L)}
    floor_cats = []
    if args.floor_count > 0:
        fcm = load_map(args.floor_map) if args.floor_map else cm
        if args.floor_cats:
            floor_cats = [c.strip() for c in args.floor_cats.split(",") if c.strip()]
        else:
            # untargeted base channel = every category at weight 1.0 (falls back to all cats
            # when no weights were given, e.g. a balanced map used as its own floor).
            floor_cats = [c for c in fcm["categories"].keys() if float(weights.get(c, 1.0)) == 1.0]
        if not floor_cats:
            raise SystemExit("--floor-count set but no floor categories resolved (all cats weighted?); "
                             "pass --floor-cats explicitly.")
        fL = int(fcm["metadata"]["num_layers"])
        fE = int(fcm["metadata"]["num_experts"])
        if (fL, fE) != (L, E):
            raise SystemExit(f"floor-map dims ({fL},{fE}) != competence-map ({L},{E})")
        fraw = raw_per_cat_layer(fcm, floor_cats, L, E, args.score)
        # base importance = sum over floor cats of rank-normalized score (untargeted, robust).
        for li in range(L):
            base = defaultdict(float)
            for c in floor_cats:
                rn = rank_normalize(fraw[c][li], E)
                for e in range(E):
                    base[e] += rn[e]
            # top-F by base importance (desc), deterministic tie-break on eid.
            top = sorted(range(E), key=lambda e: (-base[e], e))[:args.floor_count]
            protected[li] = set(top)

    # ── drop: bottom drop-count of the NON-protected experts by targeting importance ─
    drop_map = {}
    for li in range(L):
        candidates = [e for e in range(E) if e not in protected[li]]
        ranked = sorted(candidates, key=lambda e: (importance[li].get(e, 0.0), e))  # least first
        drop_map[str(li)] = sorted(ranked[:args.drop_count])
        assert len(drop_map[str(li)]) == args.drop_count
        assert not (set(drop_map[str(li)]) & protected[li]), f"L{li}: floor expert dropped"

    if args.mtp_strategy == "global":
        g = {e: aggregate(global_imp.get(e, []), "sum") for e in range(E)}
        ranked = sorted(range(E), key=lambda e: (g[e], e))
        drop_map["mtp"] = sorted(ranked[:args.drop_count])
    elif args.mtp_strategy == "layer0":
        drop_map["mtp"] = list(drop_map["0"])
    # "none" -> omit; dropper defaults MTP to layer 0

    with open(args.output, "w") as f:
        json.dump(drop_map, f)

    # ── provenance report ─────────────────────────────────────────────────────────
    print(f"[make_drop_map] competence_map={args.competence_map}")
    print(f"[make_drop_map] L={L} E={E} score={args.score} agg={args.agg} categories={len(cats)}")
    if weighted:
        wtxt = ", ".join(f"{c}={w[c]:g}" for c in cats if w[c] != 1.0) or "(all 1.0)"
        print(f"[make_drop_map] weighted agg: normalize=rank01  weights: {wtxt}")
    if args.floor_count > 0:
        print(f"[make_drop_map] floor: protect top-{args.floor_count}/layer by base channel "
              f"{floor_cats} from {args.floor_map or 'competence-map'}")
    print(f"[make_drop_map] drop {args.drop_count}/layer -> keep {keep} "
          f"({100*args.drop_count/E:.1f}% dropped)")
    print(f"[make_drop_map] mtp_strategy={args.mtp_strategy}"
          f"{' (drop '+str(len(drop_map['mtp']))+')' if 'mtp' in drop_map else ' (omitted -> layer0 default)'}")
    # overlap sanity: how stable is the drop set across layers?
    from collections import Counter
    c = Counter()
    for li in range(L):
        c.update(drop_map[str(li)])
    always = sum(1 for e in range(E) if c[e] == L)
    never = sum(1 for e in range(E) if c[e] == 0)
    print(f"[make_drop_map] experts dropped in ALL {L} layers: {always}; never dropped: {never} "
          f"(low 'always' => layer-specific specialization)")
    print(f"[make_drop_map] wrote {args.output}")


if __name__ == "__main__":
    main()
