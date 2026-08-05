#!/usr/bin/env python3
"""HF question (Qwen3.6-27B-A3B-Coder discussion #2): does rnorm track routing frequency,
and does it track the CODE-WEIGHTED frequency signal differently from the unweighted one?

WHAT CAN AND CANNOT BE ANSWERED FROM THE SHIPPED ARTIFACTS
----------------------------------------------------------
The three `*_coder*` maps -- the ones the released 27B was actually cut from -- were profiled
with `--tc-only`. Their wnorm/rnorm columns are present in the schema but IDENTICALLY ZERO
(verified: 0/92160, 0/102400, 0/112640 non-zero). So the exact question, "rnorm vs the
code-weighted signal on the shipped map", has no data behind it and no amount of analysis will
conjure one.

What DOES have data is `competence_qwen35b.json`, the balanced 8-bench baseline map, where
rnorm is populated on 73550/81920 cells. It carries humaneval + mbpp as its own categories, so
a code-vs-noncode frequency split is available on the SAME cells as rnorm. That is a proxy for
the shipped weighting (which up-weights targeted_lcb/mpe channels the balanced map does not
have) and is labelled as such throughout -- it is not the shipped map and must not be reported
as if it were.

AGGREGATION -- stated, because it changes the number
---------------------------------------------------
rnorm is per (category, layer, expert) and is an RMS over the tokens routed to that expert
(producer: rnorm = sqrt(rnsq / tc)), NOT a sum. Summing it across categories would just count
categories. It is aggregated as a tc-WEIGHTED mean, which is the pooled RMS-consistent
reduction; the unweighted mean is printed alongside so the choice is visible rather than
implicit.

Correlations are reported as SPEARMAN (rank) first, because the drop map is a per-layer
BOTTOM-K RANK selection -- rank agreement is the quantity that decides which experts die.
Pearson is printed too. Per-layer coefficients are reported as a distribution, not just the
pooled value: pooling 40 layers hides that layers differ, and the selection is per-layer.
"""
import json
import math
import os
import statistics as st
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
MAP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _HERE, "results", "competence_qwen35b.json")
CODE_CATS = {"corpus_humaneval", "corpus_mbpp"}
DROP_PER_LAYER = 72          # the shipped cut: 256 -> 184


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(a), rank(b))


def pearson(a, b):
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return float("nan")
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / math.sqrt(va * vb)


d = json.load(open(MAP))
cats = d["categories"]
allc = sorted(cats)
noncode = [c for c in allc if c not in CODE_CATS]
code = [c for c in allc if c in CODE_CATS]
layers = sorted(cats[allc[0]], key=int)
print(f"map        : {MAP.split('/')[-1]}")
print(f"categories : {len(allc)}  code={code}")
print(f"layers     : {len(layers)}   experts/layer: {len(cats[allc[0]][layers[0]])}")

rows = {}          # (layer, expert_id) -> dict of accumulators
for c in allc:
    for li in layers:
        for e in cats[c][li]:
            k = (li, e["id"])
            r = rows.setdefault(k, {"tc_all": 0.0, "tc_code": 0.0, "tc_noncode": 0.0,
                                    "rn_w": 0.0, "rn_sum": 0.0, "rn_n": 0,
                                    "wn_w": 0.0})
            tc = float(e.get("tc", 0.0))
            rn = float(e.get("rnorm", 0.0))
            wn = float(e.get("wnorm", 0.0))
            r["tc_all"] += tc
            r["tc_code" if c in CODE_CATS else "tc_noncode"] += tc
            r["rn_w"] += rn * tc
            r["wn_w"] += wn * tc
            r["rn_sum"] += rn
            r["rn_n"] += 1

for r in rows.values():
    r["rnorm"] = r["rn_w"] / r["tc_all"] if r["tc_all"] else 0.0
    r["rnorm_unw"] = r["rn_sum"] / r["rn_n"] if r["rn_n"] else 0.0
    r["wnorm"] = r["wn_w"] / r["tc_all"] if r["tc_all"] else 0.0
    # The shipped recipe up-weights the code channel; 1.5x and 2.0x are both reported because
    # the released cut used --cat-weight corpus_targeted_lcb=2.0 on a map this one lacks.
    r["w15"] = r["tc_noncode"] + 1.5 * r["tc_code"]
    r["w20"] = r["tc_noncode"] + 2.0 * r["tc_code"]

live = [r for r in rows.values() if r["tc_all"] > 0 and r["rnorm"] > 0]
print(f"cells      : {len(rows)} total, {len(live)} with BOTH tc>0 and rnorm>0 "
      f"({100.0 * len(live) / len(rows):.1f}%)")
print("             (cells with tc==0 are never routed -- no rnorm can exist for them, and\n"
      "              dropping them silently would inflate every coefficient below)")


def report(name, xk, yk):
    x = [r[xk] for r in live]
    y = [r[yk] for r in live]
    per = []
    for li in layers:
        lx = [r[xk] for (ll, _), r in rows.items() if ll == li and r["tc_all"] > 0 and r["rnorm"] > 0]
        ly = [r[yk] for (ll, _), r in rows.items() if ll == li and r["tc_all"] > 0 and r["rnorm"] > 0]
        if len(lx) > 10:
            per.append(spearman(lx, ly))
    per = [p for p in per if p == p]
    print(f"\n{name}")
    print(f"  pooled   spearman={spearman(x, y):+.4f}   pearson={pearson(x, y):+.4f}")
    if per:
        print(f"  per-layer spearman: min={min(per):+.3f}  p25={st.quantiles(per, n=4)[0]:+.3f}  "
              f"median={st.median(per):+.3f}  p75={st.quantiles(per, n=4)[2]:+.3f}  max={max(per):+.3f}")


report("rnorm  vs  UNWEIGHTED frequency (tc over all 8 benches)", "rnorm", "tc_all")
report("rnorm  vs  CODE-ONLY frequency (humaneval+mbpp)", "rnorm", "tc_code")
report("rnorm  vs  NON-CODE frequency", "rnorm", "tc_noncode")
report("rnorm  vs  code-weighted 1.5x", "rnorm", "w15")
report("rnorm  vs  code-weighted 2.0x", "rnorm", "w20")
report("wnorm  vs  UNWEIGHTED frequency  (contrast: the OTHER norm)", "wnorm", "tc_all")
report("code-weighted 1.5x  vs  UNWEIGHTED  (do the two FREQUENCY signals diverge?)",
       "w15", "tc_all")
report("code-ONLY  vs  NON-CODE frequency", "tc_code", "tc_noncode")

# --- the decision-relevant question: would ranking by rnorm cut the same experts? -----------
print("\n\nSELECTION OVERLAP -- bottom-%d per layer, which experts would actually be dropped" %
      DROP_PER_LAYER)
print("(a correlation can look high while the tail -- the only part that decides the cut -- "
      "disagrees)")
hdr = "%-26s %8s %8s" % ("bottom-72 by", "vs tc_all", "vs w15")
print(hdr)
print("-" * len(hdr))


def bottom(li, key):
    cell = [(r[key], eid) for (ll, eid), r in rows.items() if ll == li]
    cell.sort()
    return {eid for _, eid in cell[:DROP_PER_LAYER]}


for key in ("rnorm", "wnorm", "w15", "w20"):
    ov_all, ov_w15 = [], []
    for li in layers:
        b = bottom(li, key)
        ov_all.append(len(b & bottom(li, "tc_all")) / DROP_PER_LAYER)
        ov_w15.append(len(b & bottom(li, "w15")) / DROP_PER_LAYER)
    print("%-26s %7.1f%% %7.1f%%" % (key, 100 * st.mean(ov_all), 100 * st.mean(ov_w15)))
