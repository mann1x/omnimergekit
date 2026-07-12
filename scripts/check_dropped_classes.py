#!/usr/bin/env python3
"""Verify the C6v3lcb / soft2 DROPPED experts are not coder/LCB specialists.

For the experts actually dropped (per_layer_drop from the soft2 bf16 metadata),
rank every expert by wnorm within each competence class per layer, then report:
  - how many dropped experts are top-{16,40,98} by generic_code
  - how many dropped experts are top-{16,40,98} by targeted_lcb_medium_55
  - each dropped expert's ARGMAX (dominant) class
  - the worst offenders (dropped experts with the best code/LCB rank)
  - sanity: of the top-16 code/LCB experts per layer, how many were KEPT
If dropped experts are uniformly low-rank in code/LCB and none are code/LCB-
dominant, the recipe sacrificed only other-type experts -> sound.
"""
import json

MAP = "/mnt/sdc/ml/google/expert_neuron_v7_code.json"
MD = "/mnt/sdc/ml/sft_heal/dern11-soft-soft2-it/expert_drop_metadata.json"

cats = json.load(open(MAP))["categories"]
md = json.load(open(MD))
drop = md["per_layer_drop"]          # {layer_str: [expert_ids]}
keep = md["per_layer_keep"]
all_classes = sorted(cats.keys())
CODE = "generic_code"
LCB = "targeted_lcb_medium_55"
ZERO_W = {"generic_multilingual", "targeted_humaneval", "targeted_humanevalplus"}


def rank_map(cat, L):
    arr = cats[cat][str(L)]
    order = sorted(arr, key=lambda e: e["wnorm"], reverse=True)
    return {e["id"]: i + 1 for i, e in enumerate(order)}


layers = sorted(drop.keys(), key=int)
c = {k: 0 for k in (
    "total_dropped", "top16_code_drop", "top40_code_drop", "top98_code_drop",
    "top16_lcb_drop", "top40_lcb_drop", "top98_lcb_drop",
    "argmax_code_drop", "argmax_lcb_drop", "argmax_zeroW_drop",
    "top16_code_kept", "top16_lcb_kept")}
worst = []

for L in layers:
    cr = rank_map(CODE, L)
    lr = rank_map(LCB, L)
    per = {cl: rank_map(cl, L) for cl in all_classes}
    dropped = set(drop[L])
    kept = set(keep[L])
    # sanity: top-16 code/lcb experts kept?
    top16_code = [e for e, r in cr.items() if r <= 16]
    top16_lcb = [e for e, r in lr.items() if r <= 16]
    c["top16_code_kept"] += sum(1 for e in top16_code if e in kept)
    c["top16_lcb_kept"] += sum(1 for e in top16_lcb if e in kept)
    for eid in dropped:
        c["total_dropped"] += 1
        crk, lrk = cr[eid], lr[eid]
        for thr, key in ((16, "top16_code_drop"), (40, "top40_code_drop"), (98, "top98_code_drop")):
            if crk <= thr:
                c[key] += 1
        for thr, key in ((16, "top16_lcb_drop"), (40, "top40_lcb_drop"), (98, "top98_lcb_drop")):
            if lrk <= thr:
                c[key] += 1
        am = min(all_classes, key=lambda cl: per[cl][eid])
        if am == CODE:
            c["argmax_code_drop"] += 1
        elif am == LCB:
            c["argmax_lcb_drop"] += 1
        elif am in ZERO_W:
            c["argmax_zeroW_drop"] += 1
        if crk <= 40 or lrk <= 40:
            worst.append((int(L), eid, crk, lrk, am))

print("=== dropped-expert class audit (soft2 == C6v3lcb selection) ===")
print(f"layers={len(layers)}  total_dropped={c['total_dropped']} (expect 30*30=900)")
print(json.dumps({k: v for k, v in c.items() if k != "total_dropped"}, indent=1))
print(f"\nSANITY top-16 code experts KEPT: {c['top16_code_kept']}/{16*len(layers)}")
print(f"SANITY top-16 LCB  experts KEPT: {c['top16_lcb_kept']}/{16*len(layers)}")
worst.sort(key=lambda t: min(t[2], t[3]))
print(f"\nworst dropped (best code/LCB rank), {len(worst)} with code_rank<=40 or lcb_rank<=40:")
for t in worst[:30]:
    print(f"  L{t[0]:02d} e{t[1]:03d}  code_rank={t[2]:3d}  lcb_rank={t[3]:3d}  argmax={t[4]}")
