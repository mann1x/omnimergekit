#!/usr/bin/env python3
"""Honest-trade check on v8coder_fkbroad: are any of the 30 EVICTED survivors
themselves broad code/LCB specialists? (If so, pinning would be a wash.)

A survivor evicted by --force-keep is in fkbroad-drop but NOT in C6-drop. For
each, report whether it is top-16-by-wnorm in generic_code or targeted_lcb and
whether it is BROAD (PR/len >= kept p25). Want: ~0 broad code/LCB evicted.
"""
import json

MAP = "/mnt/sdc/ml/google/expert_neuron_v7_code.json"
C6 = "/srv/ml/scripts/v7coder_C6v3lcb_drop_map.json"
FK = "/srv/ml/scripts/v8coder_fkbroad_drop_map.json"
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


def dl(path):
    m = json.load(open(path))
    m = m.get("per_layer_drop", m.get("drop", m))
    return {str(k): set(v) for k, v in m.items() if str(k).lstrip("-").isdigit()}


c6, fk = dl(C6), dl(FK)
c6keep = {L: set(range(128)) - c6[L] for L in layers}
crank = {L: rank_map(CODE, L) for L in layers}
lrank = {L: rank_map(LCB, L) for L in layers}
prf = {L: {e["id"]: pr_frac(e["neuron_act"]) for e in cats[CODE][L]} for L in layers}
prl = {L: {e["id"]: pr_frac(e["neuron_act"]) for e in cats[LCB][L]} for L in layers}


def p25(rk, pr):
    v = sorted(pr[L][e] for L in layers for e, r in rk[L].items() if r <= TOPN and e in c6keep[L])
    return v[len(v) // 4]


cp25, lp25 = p25(crank, prf), p25(lrank, prl)
evicted = [(L, e) for L in layers for e in (fk[L] - c6[L])]
broad_evicted = []
for (L, e) in evicted:
    code_broad = crank[L][e] <= TOPN and prf[L][e] >= cp25
    lcb_broad = lrank[L][e] <= TOPN and prl[L][e] >= lp25
    if code_broad or lcb_broad:
        tag = ("code" if code_broad else "") + ("+lcb" if lcb_broad else "")
        broad_evicted.append((L, e, crank[L][e], lrank[L][e], tag))
print(f"evicted survivors: {len(evicted)}")
print(f"BROAD code/LCB among evicted: {len(broad_evicted)}  (want ~0)")
for (L, e, cr, lr, tag) in broad_evicted:
    print(f"  L{int(L):02d} e{e:03d}  code_rank={cr} lcb_rank={lr}  [{tag}]")
top16_evicted = sum(1 for (L, e) in evicted if crank[L][e] <= TOPN or lrank[L][e] <= TOPN)
print(f"top16 code/LCB among evicted (incl. narrow): {top16_evicted}")
