#!/usr/bin/env python
"""Per-ARM EOG statistic vs MEASURED silent-empty count. The differential test was the
wrong grain -- this is the right one.

eog_crosscheck.py falsified the draft's hypothesis at the level of the 12-expert
differential set (P24\\P12 is LESS EOG-preferential than the pool it is drawn from, not
more). But the quantity that should drive premature stopping is not the delta -- it is how
EOG-preferential the arm's WHOLE SURVIVING KEEP SET is. An arm that keeps more of the
terminator-carrying experts should emit EOG more readily, and on MultiPL-E raw-completion
that shows up as a bare-newline completion.

So: compute mean within-layer EOG-lift rank01 over each arm's keep set, and put it next to
the empty counts measured from the actual MPE samples. Six arms qualify.

WHY ONLY SIX
  armB/armC/armG/armH MERGE experts. Their surviving tensors are linear combinations, so a
  statistic defined on BASE expert ids does not describe them. Including them would be
  comparing a subset-statistic against a merged model -- a category error. They are printed
  as EXCLUDED, never silently dropped.

WHAT WOULD FALSIFY THE MECHANISM
  n=6 is small and this is observational: the arms were not designed to vary EOG content.
  A rank correlation near zero, or one driven entirely by armI, means the EOG map does not
  explain the empty rate and the story stops here. Spearman + the leave-one-out range are
  both printed so a single-point artefact cannot hide.
"""
import json
import os

import numpy as np

RECIPE = "/srv/ml/repos/omnimergekit/recipes/qwen3_6_35b_a3b_prune"
SHIPPED = f"{RECIPE}/results/drop_map_184e_coder_lcbmpe.json"
H = "/mnt/sdc/ream-work/hybrid_maps"
RNORM = f"{RECIPE}/drop_map_184e_coder_lcbmpe_rnorm.json"

# Silent-empty counts measured from the MPE sample files (n=300 per arm), 2026-08-20 audit.
EMPTIES = {"base256e": 9, "armD": 4, "armE": 13, "armF": 5, "armI p12": 18, "armJ p24": 9}
MERGED_EXCLUDED = {"armB": 11, "armC": None, "armG": 0, "armH": 10}


def main():
    m = json.load(open("/mnt/sdc/ream-work/eog_emit_map_qwen36.json"))
    E, nL = m["n_experts"], m["n_layers"]
    ew = np.array(m["emit_weight"], float)
    bw = np.array(m["bg_weight"], float)
    a = 1.0
    es = (ew + a) / (ew.sum(1, keepdims=True) + a * E)
    bs = (bw + a) / (bw.sum(1, keepdims=True) + a * E)
    lift = es / bs
    rank01 = np.argsort(np.argsort(lift, axis=1), axis=1) / (E - 1.0)

    ship = json.load(open(SHIPPED))
    layers = sorted(int(k) for k in ship if k.isdigit())
    allE = set(range(E))

    def keepsets_from_dropmap(path):
        d = json.load(open(path))
        return [sorted(allE - set(int(x) for x in d[str(L)])) for L in layers]

    arms = {
        "base256e": [sorted(allE) for _ in layers],
        "armD": keepsets_from_dropmap(SHIPPED),
        "armF": keepsets_from_dropmap(RNORM) if os.path.exists(RNORM) else None,
        "armI p12": keepsets_from_dropmap(os.path.join(H, "drop_map_184e_hybrid_p12.json")),
        "armJ p24": keepsets_from_dropmap(os.path.join(H, "drop_map_184e_hybrid_p24.json")),
    }
    # armE = top-184 by the dumped REAP saliency (gate-1-verified == armE's built keep set)
    reap = {int(k): v for k, v in json.load(open("/mnt/sdc/ream-work/reap_saliency.json")).items()}
    arms["armE"] = [sorted(sorted(range(E), key=lambda e: -reap[L][e])[:184]) for L in layers]

    print("EXCLUDED (merged experts -- base-id statistic does not describe them): %s"
          % ", ".join("%s(empties=%s)" % (k, v) for k, v in MERGED_EXCLUDED.items()))
    print("\n%-10s %7s %10s %10s   %s" % ("arm", "n_keep", "meanRank", "empties/300", "note"))
    xs, ys, labs = [], [], []
    for name in ("base256e", "armD", "armF", "armE", "armI p12", "armJ p24"):
        S = arms.get(name)
        if S is None:
            print("%-10s   (drop map not on disk -- SKIPPED, not assumed)" % name)
            continue
        v = np.mean([rank01[i, e] for i, s in enumerate(S) for e in s])
        n = len(S[0])
        emp = EMPTIES[name]
        print("%-10s %7d %10.4f %10d" % (name, n, v, emp))
        xs.append(v)
        ys.append(emp)
        labs.append(name)

    xs, ys = np.array(xs), np.array(ys, float)

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    rho = spearman(xs, ys)
    print("\nSpearman rho(mean EOG-lift rank of keep set, silent-empty count) = %+.3f  (n=%d)"
          % (rho, len(xs)))
    loo = []
    for i in range(len(xs)):
        k = [j for j in range(len(xs)) if j != i]
        loo.append((labs[i], spearman(xs[k], ys[k])))
    print("leave-one-out rho: %s" % "  ".join("-%s=%+.2f" % (l.split()[0], r) for l, r in loo))
    lo = min(r for _, r in loo)
    hi = max(r for _, r in loo)
    print("  LOO range [%+.2f, %+.2f]" % (lo, hi))
    if rho > 0.6 and lo > 0.3:
        v = ("SUPPORTED and not single-point: arms that keep more EOG-preferential experts "
             "emit more premature terminators.")
    elif rho > 0.6:
        v = ("DIRECTIONALLY SUPPORTED but FRAGILE: dropping one arm collapses it (LOO min "
             "%+.2f). Do not publish as a mechanism." % lo)
    else:
        v = ("NOT SUPPORTED: the EOG emit-map does not explain the cross-arm empty rate. "
             "The silent-empty channel has some other cause.")
    print("=> %s" % v)

    # the un-asked-for but decisive control: is the whole DROP set EOG-enriched vs chance?
    drop = [sorted(set(int(x) for x in ship[str(L)])) for L in layers]
    dv = np.mean([rank01[i, e] for i, s in enumerate(drop) for e in s])
    rng = np.random.default_rng(0)
    null = np.array([np.mean([rng.choice(rank01[i], size=72, replace=False).mean()
                              for i in range(nL)]) for _ in range(4000)])
    print("\npublished-cut DROP set mean EOG rank = %.4f ; random-72 null mean %.4f sd %.4f "
          "; p(one-sided) = %.4f" % (dv, null.mean(), null.std(), float((null >= dv).mean())))


if __name__ == "__main__":
    main()
