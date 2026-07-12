#!/usr/bin/env python3
"""Emit the --force-keep string for the BROAD code/LCB losses (the experts the
weight/floor levers cannot rescue). These are top-16-by-wnorm in generic_code or
targeted_lcb_medium_55, dropped in C6v3lcb, with neuron participation PR/len >=
the 25th percentile of the kept specialists (i.e. genuine broad contributors,
not outlier-spikes). Writes 'L:e,...' to stdout + a file.
"""
import json

MAP = "/mnt/sdc/ml/google/expert_neuron_v7_code.json"
C6 = "/srv/ml/scripts/v7coder_C6v3lcb_drop_map.json"
OUTF = "/srv/ml/scripts/broad_fk.txt"
TOPN = 16
cats = json.load(open(MAP))["categories"]
CODE, LCB = "generic_code", "targeted_lcb_medium_55"
layers = [str(i) for i in range(30)]


def pr_frac(na):
    s = sum(na)
    s2 = sum(x * x for x in na)
    return (s * s / s2) / len(na) if s2 > 0 else 0.0


def rank_map(cat, L):
    order = sorted(cats[cat][L], key=lambda e: e["wnorm"], reverse=True)
    return {e["id"]: i + 1 for i, e in enumerate(order)}


cd = {str(k): set(v) for k, v in json.load(open(C6)).items() if str(k).lstrip("-").isdigit()}
keep = {L: set(range(128)) - cd[L] for L in layers}
crank = {L: rank_map(CODE, L) for L in layers}
lrank = {L: rank_map(LCB, L) for L in layers}
prf = {L: {e["id"]: pr_frac(e["neuron_act"]) for e in cats[CODE][L]} for L in layers}
prl = {L: {e["id"]: pr_frac(e["neuron_act"]) for e in cats[LCB][L]} for L in layers}


def p25(rk, pr):
    v = sorted(pr[L][e] for L in layers for e, r in rk[L].items() if r <= TOPN and e in keep[L])
    return v[len(v) // 4]


cp25, lp25 = p25(crank, prf), p25(lrank, prl)
pins = set()
for L in layers:
    for e in cd[L]:
        if (crank[L][e] <= TOPN and prf[L][e] >= cp25) or (lrank[L][e] <= TOPN and prl[L][e] >= lp25):
            pins.add((int(L), e))
s = ",".join(f"{L}:{e}" for L, e in sorted(pins))
open(OUTF, "w").write(s)
print(f"# {len(pins)} broad code/LCB pins -> {OUTF}")
print(s)
