#!/usr/bin/env python
"""Does the p=24 hybrid rescue EOG-carrying experts that p=12 leaves dropped?

THE HYPOTHESIS (from the MPE audit): armI (p=12) emits a bare newline and stops on 18/300
MultiPL-E problems against a 1-4 replicate band; armJ (p=24) halves that to 9 and scores
better everywhere but one LCB problem. If the extra 12 experts/layer that ONLY p=24 rescues
are disproportionately terminator-carrying, that explains the silent-empty spike and the
non-monotonic dose at once -- and the fix is a --force-keep pin, not a dose change.

SET ALGEBRA (verified against the built maps by hybrid_diff.py, not assumed):
  Kbase = our 184 keeps        Dbase = our 72 drops
  P12 subset P24 subset Dbase  (protected: rescued from the drop set, REAP-ranked)
  E12 subset E24 subset Kbase  (evicted: our own lowest-ranked keeps)
  TARGET  = P24 \\ P12   (12/layer)  -- dropped by p12, rescued by p24
  POOL    = Dbase \\ P12 (60/layer)  -- everything p12 leaves dropped; TARGET's null universe
  CTRL_EV = E24 \\ E12   (12/layer)  -- evicted ONLY at p24. If EOG-heavy, p24 should be
                                        WORSE, not better; a null here is a real control.

WHY A PERMUTATION NULL AND NOT A t-TEST
  TARGET is not a random subset of POOL -- it is the top-12 of POOL by REAP saliency. So the
  question is precisely "is the REAP-ranked slice EOG-enriched relative to an arbitrary slice
  of the same pool, in the same layers?" That is a permutation question. Ranks are computed
  WITHIN layer, so a layer that routes EOG diffusely cannot dominate the statistic.

READ THE SIGN CAREFULLY
  rank01 = 1.0 means TOP of the layer by EOG lift. Hypothesis predicts
  mean_rank01(TARGET) > mean_rank01(POOL) = ~0.5.
"""
import argparse
import json
import os

import numpy as np

RECIPE = "/srv/ml/repos/omnimergekit/recipes/qwen3_6_35b_a3b_prune"
SHIPPED = f"{RECIPE}/results/drop_map_184e_coder_lcbmpe.json"
H = "/mnt/sdc/ream-work/hybrid_maps"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-map", default="/mnt/sdc/ream-work/eog_emit_map_qwen36.json")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--topk", type=int, default=8, help="top-K per layer for the census")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = json.load(open(args.emit_map))
    E, nL = m["n_experts"], m["n_layers"]
    ew = np.array(m["emit_weight"], dtype=np.float64)
    bw = np.array(m["bg_weight"], dtype=np.float64)
    ec = np.array(m["emit_count"], dtype=np.float64)
    print("emit map: layers=%d experts=%d emit_positions=%d (of %d)"
          % (nL, E, m["n_emit_positions"], m["n_positions"]))
    print("per-expert emit routings: min=%.0f p50=%.0f max=%.0f"
          % (ec.min(), np.median(ec), ec.max()))

    # Laplace-smoothed shares -> lift. Smoothing matters: an expert never routed at an emit
    # position has emit share 0 and an undefined lift, and there are only ~1.5k emit rows.
    a = 1.0
    es = (ew + a) / (ew.sum(1, keepdims=True) + a * E)
    bs = (bw + a) / (bw.sum(1, keepdims=True) + a * E)
    lift = es / bs

    # within-layer rank01 of lift: 1.0 = most EOG-preferential expert in that layer
    order = np.argsort(np.argsort(lift, axis=1), axis=1)
    rank01 = order / (E - 1.0)

    ship = json.load(open(SHIPPED))
    d12 = json.load(open(os.path.join(H, "drop_map_184e_hybrid_p12.json")))
    d24 = json.load(open(os.path.join(H, "drop_map_184e_hybrid_p24.json")))
    layers = sorted(int(k) for k in ship if k.isdigit())
    assert len(layers) == nL, (len(layers), nL)

    allE = set(range(E))
    TARGET, POOL, CTRL_EV, DROP, KEEP, P12 = [], [], [], [], [], []
    for L in layers:
        k = str(L)
        dbase = set(int(x) for x in ship[k])
        kbase = allE - dbase
        keep12 = allE - set(int(x) for x in d12[k])
        keep24 = allE - set(int(x) for x in d24[k])
        p12, e12 = keep12 - kbase, kbase - keep12
        p24, e24 = keep24 - kbase, kbase - keep24
        TARGET.append(sorted(p24 - p12))
        POOL.append(sorted(dbase - p12))
        CTRL_EV.append(sorted(e24 - e12))
        DROP.append(sorted(dbase))
        KEEP.append(sorted(kbase))
        P12.append(sorted(p12))
    for nm, S in (("TARGET", TARGET), ("POOL", POOL), ("CTRL_EV", CTRL_EV)):
        sz = set(len(x) for x in S)
        print("  %-8s %s per layer" % (nm, sz))

    def mrank(sets):
        v = [rank01[i, e] for i, S in enumerate(sets) for e in S]
        return float(np.mean(v)), len(v)

    print("\n=== Q0: is EOG capability in the published cut's DROP set at all? ===")
    mk, nk = mrank(KEEP)
    md, nd = mrank(DROP)
    print("  mean EOG-lift rank01   our KEEP (184/layer) = %.4f  (n=%d)" % (mk, nk))
    print("                         our DROP  (72/layer) = %.4f  (n=%d)" % (md, nd))
    print("  => %s" % ("the cut RETAINS the EOG-preferential experts (drop set is below "
                       "average)" if md < mk else
                       "the cut DROPS EOG-preferential experts -- a force-keep target"))

    # top-K census, T202 style
    print("\n=== top-%d EOG experts per layer vs the published drop set ===" % args.topk)
    tot = indrop = 0
    for i, L in enumerate(layers):
        top = set(np.argsort(-lift[i])[:args.topk].tolist())
        tot += len(top)
        indrop += len(top & set(DROP[i]))
    print("  %d/%d top-%d EOG experts are DROPPED by the published cut (%.1f%%; "
          "chance = %.1f%%)" % (indrop, tot, args.topk, 100.0 * indrop / tot,
                                100.0 * 72 / E))

    print("\n=== Q1 (DECISIVE): is TARGET = P24\\P12 EOG-enriched within POOL? ===")
    mt, nt = mrank(TARGET)
    mp, npool = mrank(POOL)
    rng = np.random.default_rng(args.seed)
    pool_r = [np.array([rank01[i, e] for e in POOL[i]]) for i in range(nL)]
    kk = len(TARGET[0])
    null = np.empty(args.iters)
    for t in range(args.iters):
        s = 0.0
        for i in range(nL):
            s += rng.choice(pool_r[i], size=kk, replace=False).sum()
        null[t] = s / (kk * nL)
    p_hi = float((null >= mt).mean())
    print("  mean rank01  TARGET (12/layer, REAP-top of POOL) = %.4f  (n=%d)" % (mt, nt))
    print("               POOL   (60/layer, all p12 leaves dropped) = %.4f (n=%d)"
          % (mp, npool))
    print("               permutation null: mean=%.4f sd=%.4f  p(one-sided, TARGET higher)=%.4f"
          % (null.mean(), null.std(), p_hi))
    print("  => %s" % ("ENRICHED: p=24 specifically rescues EOG-preferential experts that "
                       "p=12 leaves dropped. Explains the empty-rate gap; --force-keep is "
                       "the fix." if p_hi < 0.05 else
                       "NOT ENRICHED: the p12->p24 empty-rate difference is NOT explained "
                       "by terminator experts. The hypothesis in the draft is WRONG as "
                       "stated and must not be published."))

    print("\n=== Q2 (control): CTRL_EV = E24\\E12, evicted ONLY at the higher dose ===")
    mc, nc = mrank(CTRL_EV)
    kpool_r = [np.array([rank01[i, e] for e in KEEP[i]]) for i in range(nL)]
    kc = len(CTRL_EV[0])
    null2 = np.empty(args.iters)
    for t in range(args.iters):
        s = 0.0
        for i in range(nL):
            s += rng.choice(kpool_r[i], size=kc, replace=False).sum()
        null2[t] = s / (kc * nL)
    p_lo = float((null2 <= mc).mean())
    print("  mean rank01 CTRL_EV = %.4f (n=%d) vs KEEP-pool null mean %.4f  "
          "p(one-sided, lower)=%.4f" % (mc, nc, null2.mean(), p_lo))
    print("  (p=24 evicting EOG-poor experts would be consistent; evicting EOG-RICH ones "
           "would contradict p=24 being the better arm)")

    print("\n=== Q3: P12 itself (rescued by BOTH doses) ===")
    m1, n1 = mrank(P12)
    print("  mean rank01 P12 = %.4f (n=%d)" % (m1, n1))

    out = "/mnt/sdc/ream-work/eog_crosscheck_result.json"
    json.dump({"keep": mk, "drop": md, "target": mt, "pool": mp, "ctrl_ev": mc, "p12": m1,
               "p_target_enriched": p_hi, "p_ctrl_lower": p_lo,
               "topk_in_drop": [indrop, tot, args.topk]}, open(out, "w"))
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
