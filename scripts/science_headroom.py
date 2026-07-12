#!/usr/bin/env python3
"""science_headroom.py — size the "more science" lever for v8b.

(1) How many loop-safe science/ml experts (dominant class science|multilingual,
    agentic_eog token-106 rank > 16) are NOT already in v8b-safe's keep set?
    => the headroom for restoring more science without reopening loops.
(2) Rank the 30 fkbroad code/LCB force-keep pins by code value (generic_code +
    targeted_lcb rank); weakest-code pins are the cheapest to un-pin to free slots.

Usage: science_headroom.py <keepmeta.json> <code_map.json> <agentic_t106_map.json> <pins_str>
"""
import collections
import json
import sys

keepmeta_p, code_p, ag_p, pins_s = sys.argv[1:5]
NL = 30
SCI = {"generic_science", "generic_multilingual"}


def load_cats(p):
    d = json.load(open(p))
    return d.get("categories", d)


code = load_cats(code_p)
ag = load_cats(ag_p)
agcat = None
for c in ag:
    cl = c.lower()
    if "agentic" in cl or "eog" in cl or "t106" in cl:
        agcat = c
        break
if agcat is None and len(ag) == 1:
    agcat = list(ag)[0]
allc = list(code.keys())


def lr(cats, cat, L):
    row = cats[cat][str(L)]
    order = sorted(row, key=lambda e: float(e["wnorm"]), reverse=True)
    return {e["id"]: i + 1 for i, e in enumerate(order)}


def dom(L):
    best = {}
    for c in allc:
        for e in code[c][str(L)]:
            w = float(e["wnorm"])
            eid = e["id"]
            if eid not in best or w > best[eid][1]:
                best[eid] = (c, w)
    return {eid: cv[0] for eid, cv in best.items()}


km = json.load(open(keepmeta_p))
keep = {int(L): set(v) for L, v in km["keep"].items()}
pins = collections.defaultdict(set)
for tok in pins_s.split(","):
    L, e = tok.split(":")
    pins[int(L)].add(int(e))

# (1) headroom: loop-safe science NOT already kept
cands = []
for L in range(NL):
    arank = lr(ag, agcat, L)
    domc = dom(L)
    srank = lr(code, "generic_science", L) if "generic_science" in code else {}
    for e in range(128):
        if e in keep[L]:
            continue
        if domc.get(e) in SCI and arank.get(e, 999) > 16:
            cands.append((L, e, arank.get(e, 999), srank.get(e, 999), domc.get(e)))
byL = collections.Counter(L for (L, _, _, _, _) in cands)
print(f"== loop-safe science/ml NOT in v8b-safe keep (HEADROOM): {len(cands)} experts")
print(f"   layers with headroom: {len([L for L in byL if byL[L]>0])}/30")
print(f"   per-layer headroom: {dict(sorted(byL.items()))}")
# best science candidate per layer (by science rank)
print("   top headroom candidates (L:e agRank sciRank dom):")
for r in sorted(cands, key=lambda x: x[3])[:20]:
    print(f"     L{r[0]:>2} e{r[1]:>3}  ag={r[2]:>3} sci={r[3]:>3} dom={r[4]}")

# (2) weakest code pins (cheapest to un-pin)
crank = {L: lr(code, "generic_code", L) for L in range(NL)}
lrank = {L: lr(code, "targeted_lcb_medium_55", L) for L in range(NL)}
pinrows = []
for L in pins:
    for e in pins[L]:
        cv = crank[L].get(e, 999) + lrank[L].get(e, 999)
        pinrows.append((cv, L, e, crank[L].get(e, 999), lrank[L].get(e, 999), byL.get(L, 0)))
pinrows.sort(reverse=True)  # weakest code first
print("\n== 30 code pins ranked by code value (WEAKEST code first); cheapest to un-pin")
print(f"{'cv':>5} {'L:e':>7} {'codeRank':>9} {'lcbRank':>8} {'layerHeadroom':>14}")
for (cv, L, e, cr, lr_, hr) in pinrows:
    print(f"{cv:>5} {L:>3}:{e:<3} {cr:>9} {lr_:>8} {hr:>14}")
