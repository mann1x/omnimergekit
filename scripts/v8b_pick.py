#!/usr/bin/env python3
"""Pick the v8b winner: diff each candidate drop-map vs fkbroad and score
science/multilingual restored vs code/lcb lost, using the competence map ranks.

Usage: v8b_pick.py <code_map.json> <fkbroad_drop.json> <v7coder_drop.json> <cand1.json> [cand2 ...]

A candidate is good when its swapped-IN experts (dropped by fkbroad, kept now)
rank high in generic_science/multilingual, while its swapped-OUT experts
(kept by fkbroad, dropped now) do NOT rank high in generic_code/targeted_lcb.
"""
import json
import os
import sys

code_map = sys.argv[1]
fkbroad = sys.argv[2]
v7coder = sys.argv[3]
cands = sys.argv[4:]

cats = json.load(open(code_map))["categories"]
NL = 30
NE = 128
K = 30  # top-K per layer (= the per-layer drop budget)


def load_drop(p):
    m = json.load(open(p))
    return {int(k): set(v) for k, v in (m.items() if isinstance(m, dict) else enumerate(m))}


def rank_topk(cat, layer, k=K):
    """Set of expert ids in the top-k by wnorm for (category, layer)."""
    row = cats[cat][str(layer)]
    order = sorted(row, key=lambda e: float(e["wnorm"]), reverse=True)
    return {e["id"] for e in order[:k]}


# dominant-class specialist: for (layer, expert) the argmax class among all 9
SIG_SCI = {"generic_science", "generic_multilingual"}
SIG_CODE = {"generic_code", "targeted_lcb_medium_55"}
ALLCATS = list(cats.keys())


def dom_class():
    """{layer: {expert_id: argmax_category}} over all 9 classes by wnorm."""
    out = {}
    for L in range(NL):
        best = {}
        for c in ALLCATS:
            for e in cats[c][str(L)]:
                w = float(e["wnorm"])
                eid = e["id"]
                if eid not in best or w > best[eid][1]:
                    best[eid] = (c, w)
        out[L] = {eid: cv[0] for eid, cv in best.items()}
    return out


DOM = dom_class()
fk = load_drop(fkbroad)
v7 = load_drop(v7coder)

print(f"{'cand':<8} {'swap':>5} {'in_sciSpec':>10} {'out_codeSpec':>12} "
      f"{'in_top30sm':>10} {'out_top30cl':>11} {'in∈v7':>6}")
print("-" * 70)
TOP = {c: {L: rank_topk(c, L) for L in range(NL)}
       for c in ("generic_science", "generic_multilingual",
                 "generic_code", "targeted_lcb_medium_55")}
rows = []
for cp in cands:
    cand = load_drop(cp)
    swap = 0
    in_sci_spec = out_code_spec = 0      # dominant-class specialists (decisive)
    in_top30sm = out_top30cl = 0         # coarse top-30 view (for reference)
    in_in_v7 = 0
    for L in range(NL):
        si = fk[L] - cand[L]   # restored (kept now, fkbroad dropped)
        so = cand[L] - fk[L]   # cost     (dropped now, fkbroad kept)
        swap += len(si)
        for e in si:
            if DOM[L].get(e) in SIG_SCI:
                in_sci_spec += 1
            if e in TOP["generic_science"][L] or e in TOP["generic_multilingual"][L]:
                in_top30sm += 1
            if e not in v7[L]:
                in_in_v7 += 1
        for e in so:
            if DOM[L].get(e) in SIG_CODE:
                out_code_spec += 1
            if e in TOP["generic_code"][L] or e in TOP["targeted_lcb_medium_55"][L]:
                out_top30cl += 1
    name = os.path.basename(cp).replace("v8b_", "").replace("_drop_map.json", "")
    rows.append((name, swap, in_sci_spec, out_code_spec))
    print(f"{name:<8} {swap:>5} {in_sci_spec:>10} {out_code_spec:>12} "
          f"{in_top30sm:>10} {out_top30cl:>11} {in_in_v7:>6}")

print("-" * 70)
print("in_sciSpec  = swapped-IN whose DOMINANT class is science/multilingual (true science gain)")
print("out_codeSpec= swapped-OUT whose DOMINANT class is code/lcb (true code loss — want ~0)")
print("top30 cols  = coarse membership view (generalists double-count)")
# protect-code mandate: minimize out_codeSpec; among those, maximize in_sciSpec
best = min(rows, key=lambda r: (r[3], -r[2]))
print(f"\nRECOMMEND (protect-code): {best[0]}  "
      f"(swap={best[1]} sci_gain={best[2]} code_loss={best[3]})")
