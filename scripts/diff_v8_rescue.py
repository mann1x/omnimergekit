#!/usr/bin/env python3
"""Rescue analysis: how well does each candidate v8 drop map recover the BROAD
code/LCB specialists that C6v3lcb dropped, and what does it evict in exchange.

Usage: diff_v8_rescue.py <candidate_map.json> [<candidate2.json> ...]
Reference = C6v3lcb. For each candidate vs C6:
  - total swaps (experts swapped in/out across 30 layers)
  - BROAD code/LCB losses rescued (of the 41 from the outlier check)
  - all top-16 code/LCB C6-drops now kept
  - swapped-OUT dominant-class histogram (want generalist/zero-weight, NOT code/lcb)
  - swapped-IN dominant-class histogram (want code/lcb)
"""
import json
import sys
from collections import Counter

MAP = "/mnt/sdc/ml/google/expert_neuron_v7_code.json"
C6 = "/srv/ml/scripts/v7coder_C6v3lcb_drop_map.json"
TOPN = 16
cats = json.load(open(MAP))["categories"]
all_classes = sorted(cats.keys())
CODE, LCB = "generic_code", "targeted_lcb_medium_55"
layers = [str(i) for i in range(30)]


def pr_frac(na):
    s = sum(na)
    s2 = sum(x * x for x in na)
    return (s * s / s2) / len(na) if s2 > 0 else 0.0


def rank_map(cat, L):
    arr = cats[cat][L]
    order = sorted(arr, key=lambda e: e["wnorm"], reverse=True)
    return {e["id"]: i + 1 for i, e in enumerate(order)}


def dl(path):
    m = json.load(open(path))
    m = m.get("per_layer_drop", m.get("drop", m))
    return {str(k): set(v) for k, v in m.items() if str(k).lstrip("-").isdigit()}


prf = {L: {e["id"]: pr_frac(e["neuron_act"]) for e in cats[CODE][L]} for L in layers}
prf_lcb = {L: {e["id"]: pr_frac(e["neuron_act"]) for e in cats[LCB][L]} for L in layers}
crank = {L: rank_map(CODE, L) for L in layers}
lrank = {L: rank_map(LCB, L) for L in layers}
argmax = {L: {} for L in layers}
for L in layers:
    pr = {c: rank_map(c, L) for c in all_classes}
    for e in cats[CODE][L]:
        eid = e["id"]
        argmax[L][eid] = min(all_classes, key=lambda c: pr[c][eid])

c6drop = dl(C6)
c6keep = {L: set(range(128)) - c6drop[L] for L in layers}

# broad-loss set: top-16 code or lcb, dropped in C6, broad (PR>=kept p25 of that class)
def kept_p25(rankmap, prmap):
    vals = []
    for L in layers:
        for eid, r in rankmap[L].items():
            if r <= TOPN and eid in c6keep[L]:
                vals.append(prmap[L][eid])
    vals.sort()
    return vals[len(vals) // 4]


cp25 = kept_p25(crank, prf)
lp25 = kept_p25(lrank, prf_lcb)
broad = set()
top16_drop = set()
for L in layers:
    for eid in c6drop[L]:
        if crank[L][eid] <= TOPN:
            top16_drop.add((L, eid))
            if prf[L][eid] >= cp25:
                broad.add((L, eid))
        if lrank[L][eid] <= TOPN:
            top16_drop.add((L, eid))
            if prf_lcb[L][eid] >= lp25:
                broad.add((L, eid))
print(f"reference C6v3lcb: top16 code/lcb dropped={len(top16_drop)}  BROAD losses={len(broad)}")

for path in sys.argv[1:]:
    cd = dl(path)
    ckeep = {L: set(range(128)) - cd[L] for L in layers}
    swapped_in = [(L, e) for L in layers for e in (ckeep[L] - c6keep[L])]
    swapped_out = [(L, e) for L in layers for e in (cd[L] - c6drop[L])]
    broad_resc = sum(1 for (L, e) in broad if e in ckeep[L])
    top16_resc = sum(1 for (L, e) in top16_drop if e in ckeep[L])
    out_hist = Counter(argmax[L][e] for (L, e) in swapped_out)
    in_hist = Counter(argmax[L][e] for (L, e) in swapped_in)
    name = path.split("/")[-1].replace("_drop_map.json", "")
    print(f"\n### {name}")
    print(f"  total swaps: {len(swapped_in)} (in) / {len(swapped_out)} (out)")
    print(f"  BROAD losses rescued: {broad_resc}/{len(broad)}   top16 code/lcb rescued: {top16_resc}/{len(top16_drop)}")
    print(f"  swapped-IN  dominant-class: {dict(in_hist.most_common())}")
    print(f"  swapped-OUT dominant-class: {dict(out_hist.most_common())}")
