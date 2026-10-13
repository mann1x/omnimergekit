#!/usr/bin/env python3
"""eog_map_overlap.py — split-half stability of the LIFT pin selection.

The t106 map rests on 71 model-owned positions and touches only ~20 of 128 experts per
layer. That concentration is either real signal or sampling noise, and the difference
decides whether a force-keep pin drawn from it means anything. This builds the SAME
selection the ranker would make, on two disjoint halves of the corpus, and reports how
often they pick the same expert.

The background arm is shared between halves on purpose: it is the population reference
(1.95M positions), it is not the thin thing, and splitting it would only add noise to the
denominator. What is under test is the stability of the emit-side numerator.

Read the result against the CHANCE line, not against 100%. Picking k of 128 at random
agrees k/128 of the time; for k=4 that is 3.1%.
"""
from __future__ import annotations

import argparse
import json


def load(path, key):
    d = json.load(open(path))
    cats = d.get("categories", {})
    if key not in cats:
        raise SystemExit("%s has no category %r (has %s)" % (path, key, list(cats)))
    return cats[key]


def shares(cat, field):
    out = {}
    for layer, rows in cat.items():
        vals = [float(r.get(field, 0.0)) for r in rows]
        tot = sum(vals)
        out[layer] = [v / tot if tot > 0 else 0.0 for v in vals]
    return out


def pick(emit, bg_share, field, k, min_tc):
    """Reproduce rank_agentic_eog_lift's per-layer top-k selection."""
    se = shares(emit, field)
    out = {}
    for layer, rows in emit.items():
        cand = []
        for e in range(len(rows)):
            tc = float(rows[e].get("tc", 0.0))
            if tc < min_tc:
                continue
            b = bg_share[layer][e]
            lift = float("inf") if b == 0 else se[layer][e] / b
            cand.append((lift, tc, e))
        cand.sort(key=lambda x: (-x[0], -x[1]))
        out[layer] = [e for _, _, e in cand[:k]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half-a", required=True)
    ap.add_argument("--half-b", required=True)
    ap.add_argument("--bg-map", required=True, help="shared background (population reference)")
    ap.add_argument("--field", default="tc")
    ap.add_argument("--min-tc", type=int, default=4,
                    help="Lower than the full-map floor: each half holds half the "
                         "positions, so the same absolute floor would be twice as strict.")
    args = ap.parse_args()

    A = load(args.half_a, "agentic_eog")
    B = load(args.half_b, "agentic_eog")
    bg = shares(load(args.bg_map, "agentic_eog_bg"), args.field)
    n_exp = len(A[next(iter(A))])

    print("split-half LIFT stability   field=%s  min_tc=%d  experts/layer=%d"
          % (args.field, args.min_tc, n_exp))
    print("%-5s %8s %9s %9s %9s" % ("k", "mean", "chance", "min", "max"))
    for k in (1, 2, 4, 8, 16):
        pa, pb = pick(A, bg, args.field, k, args.min_tc), pick(B, bg, args.field, k, args.min_tc)
        ov = []
        for layer in pa:
            if layer not in pb:
                continue
            denom = min(k, max(len(pa[layer]), 1), max(len(pb[layer]), 1))
            ov.append(len(set(pa[layer]) & set(pb[layer])) / max(denom, 1))
        if not ov:
            continue
        print("%-5d %7.1f%% %8.1f%% %8.1f%% %8.1f%%"
              % (k, 100 * sum(ov) / len(ov), 100 * k / n_exp, 100 * min(ov), 100 * max(ov)))
    print("\nRead against CHANCE. Near chance => the selection is sampling noise and no pin "
          "drawn from this map is meaningful; well above => the concentration is real and "
          "the thin position count is tolerable for a force-keep set.")


if __name__ == "__main__":
    main()
