#!/usr/bin/env python3
"""Addendum: does the AGGREGATOR decide whether the code channel survives?

The first pass used --agg sum, make_drop_map's default. But the SHIPPED coder cut used
`--agg wmax --cat-weight corpus_targeted_lcb=2.0`. Under `sum`, up-weighting 2 of 8 categories
by 1.5-2x is arithmetically almost a no-op (measured: Spearman 0.995 vs unweighted), so
reporting only the sum result would imply the recipe's weighting does nothing -- which would be
a claim about a configuration the release did not use. Measure wmax on the same cells before
saying anything about dilution.

Aggregators (make_drop_map semantics):
  sum   -- sum_c w_c * tc_c            (a category is 1/8 of the total; weighting is diluted)
  wmax  -- max_c w_c * tc_c            (a weighted category can OWN the score for an expert)
"""
import json
import math
import os
import statistics as st
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
MAP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _HERE, "results", "competence_qwen35b.json")
CODE = {"corpus_humaneval", "corpus_mbpp"}
K = 72


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return float("nan")
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / math.sqrt(va * vb)


def spearman(a, b):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[o[k]] = avg
            i = j + 1
        return r
    return pearson(rk(a), rk(b))


d = json.load(open(MAP))["categories"]
cats = sorted(d)
layers = sorted(d[cats[0]], key=int)
tc = {}          # (layer, eid) -> {cat: tc}
rn = {}
for c in cats:
    for li in layers:
        for e in d[c][li]:
            tc.setdefault((li, e["id"]), {})[c] = float(e.get("tc", 0.0))
            rn.setdefault((li, e["id"]), []).append(
                (float(e.get("rnorm", 0.0)), float(e.get("tc", 0.0))))

rnorm = {k: (sum(r * t for r, t in v) / sum(t for _, t in v)) if sum(t for _, t in v) else 0.0
         for k, v in rn.items()}


def score(k, agg, w):
    vals = [w.get(c, 1.0) * tc[k][c] for c in cats]
    return sum(vals) if agg == "sum" else max(vals)


W1, W15, W20 = {}, {c: 1.5 for c in CODE}, {c: 2.0 for c in CODE}
variants = {
    "sum  unweighted": ("sum", W1),
    "sum  code=1.5":   ("sum", W15),
    "sum  code=2.0":   ("sum", W20),
    "wmax unweighted": ("wmax", W1),
    "wmax code=1.5":   ("wmax", W15),
    "wmax code=2.0":   ("wmax", W20),
}
S = {n: {k: score(k, a, w) for k in tc} for n, (a, w) in variants.items()}

ref = "sum  unweighted"
print("Rank agreement with the UNWEIGHTED sum baseline (per-layer Spearman, median over 40L)")
print("%-18s %10s %14s" % ("variant", "spearman", "bottom-72 same"))
print("-" * 46)
for n in variants:
    per, ov = [], []
    for li in layers:
        ks = [k for k in tc if k[0] == li]
        per.append(spearman([S[n][k] for k in ks], [S[ref][k] for k in ks]))
        b1 = {k for k in sorted(ks, key=lambda k: S[n][k])[:K]}
        b0 = {k for k in sorted(ks, key=lambda k: S[ref][k])[:K]}
        ov.append(len(b1 & b0) / K)
    print("%-18s %+10.4f %13.1f%%" % (n, st.median(per), 100 * st.mean(ov)))

print("\nrnorm vs each variant (per-layer Spearman, median over 40 layers)")
for n in variants:
    per = []
    for li in layers:
        ks = [k for k in tc if k[0] == li and rnorm[k] > 0]
        if len(ks) > 10:
            per.append(spearman([rnorm[k] for k in ks], [S[n][k] for k in ks]))
    print("  %-18s %+0.4f" % (n, st.median(per)))
