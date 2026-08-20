#!/usr/bin/env python
"""Is activation-norm saliency (rnorm) a different signal from routing frequency (tc)?

This is the cheap test promised in the HF discussion. The shipped Qwen3.6 coder maps were
profiled with --tc-only, so they carry routing counts but a dead rnorm field -- which means
we had never actually MEASURED whether the two axes disagree on our own basis. A full
re-profile populates both, and both come from the SAME forward passes over the SAME corpus,
so the comparison is internal to one map and free of any cross-run confound.

Three readouts, cheapest first:

  1. Per-(category, layer) Spearman rho between tc and rnorm across all E experts. This is
     the raw question: do the two axes rank experts the same way?
  2. Keep-set overlap at the shipped cut size. Rank correlation can look low while the
     TOP-K sets still coincide (disagreement concentrated in the tail nobody keeps), and it
     is the keep set -- not the ranking -- that decides what the model becomes. So we build
     the layer importance the way make_drop_map does (rank-normalize per category, then a
     weighted aggregate) under each score and intersect the resulting keep sets.
  3. Overlap of each against the SHIPPED drop map, so the numbers are anchored to the model
     that actually exists rather than to each other only.

A high rho AND a high keep-set overlap means the tc-only shortcut cost us nothing and there
is no second model worth building. A low overlap means there is a genuinely different cut to
evaluate, and readout 2 tells us exactly how many experts per layer would change.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
import make_drop_map as mdm  # noqa: E402


def spearman(a, b):
    """Rank correlation without scipy. Average ranks so ties (common in tc) are handled."""
    def rank(x):
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(len(x), dtype=float)
        # average ranks within tie groups
        xs = x[order]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r

    ra, rb = rank(np.asarray(a, dtype=float)), rank(np.asarray(b, dtype=float))
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def layer_importance(cm, cats, L, E, score, agg, cat_weights):
    """Reproduce make_drop_map's layer scoring for one score kind -> {li: {eid: importance}}."""
    raw = mdm.raw_per_cat_layer(cm, cats, L, E, score)
    out = {}
    for li in range(L):
        per_expert = {e: [] for e in range(E)}
        for cat in cats:
            rank01 = mdm.rank_normalize(raw[cat][li], E)
            w = cat_weights.get(cat, 1.0)
            for e in range(E):
                per_expert[e].append((rank01[e], w))
        agg_out = {}
        for e in range(E):
            vals = per_expert[e]
            if agg == "wmax":
                agg_out[e] = max(v * w for v, w in vals)
            elif agg == "wsum":
                agg_out[e] = sum(v * w for v, w in vals)
            else:
                agg_out[e] = mdm.aggregate([v for v, _ in vals], agg)
        out[li] = agg_out
    return out


def keep_set(importance_layer, keep_n):
    return set(sorted(importance_layer, key=lambda e: (-importance_layer[e], e))[:keep_n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--competence-map", required=True)
    ap.add_argument("--drop-map", default=None,
                    help="shipped drop map, to anchor the overlap on the model that exists")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--agg", default="wmax", choices=mdm.RAW_AGGS + mdm.WEIGHTED_AGGS)
    ap.add_argument("--cat-weight", action="append", default=[], metavar="CAT=W")
    ap.add_argument("--out", default=None, help="write the full per-layer table as JSON")
    args = ap.parse_args()

    cm = mdm.load_map(args.competence_map)
    cats = sorted(cm["categories"].keys())
    L, E = args.layers, args.experts
    cat_weights = mdm.parse_cat_weights(args.cat_weight)

    # A --tc-only map still emits the rnorm KEY -- it is 0.0 everywhere. So probe the values,
    # not the schema: a whole layer of zeros would make every readout below meaningless
    # (spearman of a constant is nan; the keep set would fall back to the eid tie-break).
    for cat in cats:
        for li in (0, L // 2, L - 1):
            row = cm["categories"][cat][str(li)]
            if not any(float(e.get("rnorm", 0.0)) for e in row):
                raise SystemExit(
                    f"refusing: rnorm is 0.0 for all {len(row)} experts at {cat} L{li} -- this "
                    f"map was profiled with --tc-only, so there is no activation-norm axis to "
                    f"compare against. Re-profile without --tc-only first.")
    print(f"map: {args.competence_map}")
    print(f"categories ({len(cats)}): {', '.join(cats)}")
    print(f"layers={L} experts={E} agg={args.agg} cat_weights={cat_weights or '{}'}\n")

    # ---- readout 1: raw per-(cat, layer) rank correlation -------------------------------
    print("=" * 78)
    print("1. Spearman rho(tc, rnorm) across experts, per category (over all layers)")
    print("=" * 78)
    print(f"{'category':<26} {'mean rho':>9} {'min':>8} {'max':>8}  {'n_layers':>8}")
    per_cat_rho = {}
    for cat in cats:
        rhos = []
        for li in range(L):
            row = cm["categories"][cat][str(li)]
            tc = [float(e.get("tc", 0.0)) for e in row]
            rn = [float(e.get("rnorm", 0.0)) for e in row]
            rhos.append(spearman(tc, rn))
        rhos = [r for r in rhos if not np.isnan(r)]
        per_cat_rho[cat] = rhos
        print(f"{cat:<26} {np.mean(rhos):>9.4f} {min(rhos):>8.4f} {max(rhos):>8.4f} {len(rhos):>9}")
    allr = [r for v in per_cat_rho.values() for r in v]
    print(f"\n{'ALL':<26} {np.mean(allr):>9.4f} {min(allr):>8.4f} {max(allr):>8.4f} {len(allr):>9}")

    # ---- readouts 2 + 3: keep-set overlap at the shipped cut ----------------------------
    drop_map = None
    keep_n = None
    if args.drop_map:
        with open(args.drop_map) as fh:
            drop_map = json.load(fh)
        keep_n = E - len(drop_map["0"])
        print(f"\nshipped drop map: {args.drop_map} -> keep {keep_n}/{E} per layer")
    else:
        keep_n = E * 184 // 256
        print(f"\nno --drop-map given; using keep_n={keep_n}")

    imp = {s: layer_importance(cm, cats, L, E, s, args.agg, cat_weights)
           for s in ("tc", "rnorm")}

    print("\n" + "=" * 78)
    print(f"2/3. Keep-set overlap at keep={keep_n}/{E}, per layer")
    print("=" * 78)
    hdr = f"{'layer':>5} {'tc<->rnorm':>11} {'rho':>8}"
    if drop_map:
        hdr += f" {'tc<->ship':>10} {'rnorm<->ship':>13}"
    print(hdr)

    rows, ov_tr, ov_ts, ov_rs = [], [], [], []
    for li in range(L):
        k_tc = keep_set(imp["tc"][li], keep_n)
        k_rn = keep_set(imp["rnorm"][li], keep_n)
        o_tr = len(k_tc & k_rn) / keep_n
        ov_tr.append(o_tr)
        rho_l = np.mean([spearman([float(e.get("tc", 0.0)) for e in cm["categories"][c][str(li)]],
                                  [float(e.get("rnorm", 0.0)) for e in cm["categories"][c][str(li)]])
                         for c in cats])
        line = f"{li:>5} {o_tr:>11.4f} {rho_l:>8.4f}"
        row = {"layer": li, "overlap_tc_rnorm": o_tr, "rho": float(rho_l)}
        if drop_map:
            k_sh = set(range(E)) - set(int(x) for x in drop_map[str(li)])
            o_ts, o_rs = len(k_tc & k_sh) / keep_n, len(k_rn & k_sh) / keep_n
            ov_ts.append(o_ts)
            ov_rs.append(o_rs)
            line += f" {o_ts:>10.4f} {o_rs:>13.4f}"
            row["overlap_tc_shipped"] = o_ts
            row["overlap_rnorm_shipped"] = o_rs
        rows.append(row)
        print(line)

    print("-" * 78)
    summ = f"{'MEAN':>5} {np.mean(ov_tr):>11.4f} {np.mean(allr):>8.4f}"
    if drop_map:
        summ += f" {np.mean(ov_ts):>10.4f} {np.mean(ov_rs):>13.4f}"
    print(summ)

    changed = [(1 - o) * keep_n for o in ov_tr]
    print(f"\nexperts per layer that CHANGE if you cut by rnorm instead of tc: "
          f"mean {np.mean(changed):.1f} / {keep_n}  (min {min(changed):.0f}, max {max(changed):.0f})")
    print(f"total expert swaps across {L} layers: {sum(changed):.0f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"map": args.competence_map, "drop_map": args.drop_map,
                       "categories": cats, "keep_n": keep_n, "agg": args.agg,
                       "cat_weights": cat_weights,
                       "mean_rho": float(np.mean(allr)),
                       "mean_overlap_tc_rnorm": float(np.mean(ov_tr)),
                       "mean_overlap_tc_shipped": float(np.mean(ov_ts)) if drop_map else None,
                       "mean_overlap_rnorm_shipped": float(np.mean(ov_rs)) if drop_map else None,
                       "per_layer": rows}, fh, indent=2)
        print(f"\n>>> RNORM_TC_WROTE {args.out}")
    print("\n>>> RNORM_TC_DONE")


if __name__ == "__main__":
    main()
