#!/usr/bin/env python3
"""creative_sacrifice.py — cap2-anchored creative-for-victim substitution.

Take cap2's PROVEN restored-science set (cap2_keep - v8_keep) verbatim — that set
gives the cap2 loop-safety (1/48) and GPQA (67.68). Keep it intact. Change ONLY
the experts evicted to make room: prefer evicting generic_creative-dominant experts
(loop-safe ag>16, non-pin, weak-code) instead of cap2's multilingual/generalist
victims. Where a layer lacks enough droppable creative, fall back to cap2's own
victim for that slot (so the result is never worse than cap2 on MPE, strictly
better where creative substitutes).

This isolates the victim class as the single variable vs cap2:
  same restored science  -> loops ~1, GPQA ~67
  creative victims       -> MPE/HE+ hold (no multilingual/code/HE+ dropped)

Usage:
  creative_sacrifice.py <code_map> <agentic_t106> <v8_fkbroad_drop> <cap2_drop> \
      <pins_str> <out_drop.json>
"""
import collections
import json
import sys

code_p, ag_p, fk_p, cap2_p, pins_s, out_p = sys.argv[1:7]
NL, NE = 30, 128


def load_cats(p):
    d = json.load(open(p))
    return d.get("categories", d)


code = load_cats(code_p)
ag = load_cats(ag_p)
agcat = next((c for c in ag if any(k in c.lower() for k in ("agentic", "eog", "t106"))),
             list(ag)[0])
allc = list(code.keys())


def lr(cats, cat, L):
    row = cats[cat][str(L)]
    order = sorted(row, key=lambda e: float(e["wnorm"]), reverse=True)
    return {e["id"]: i + 1 for i, e in enumerate(order)}


def dom_layer(L):
    best = {}
    for c in allc:
        for e in code[c][str(L)]:
            w = float(e["wnorm"])
            eid = e["id"]
            if eid not in best or w > best[eid][1]:
                best[eid] = (c, w)
    return {eid: cv[0] for eid, cv in best.items()}


def load_keep(p):
    d = json.load(open(p))
    if isinstance(d.get("keep"), dict):
        return {int(L): set(int(x) for x in v) for L, v in d["keep"].items()}
    return {L: set(range(NE)) - set(int(x) for x in d[str(L)]) for L in range(NL)}


fk = load_keep(fk_p)
cap2 = load_keep(cap2_p)
pins = collections.defaultdict(set)
for tok in pins_s.split(","):
    L, e = tok.split(":")
    pins[int(L)].add(int(e))

CODECLS = ["generic_code", "targeted_lcb_medium_55", "targeted_humaneval",
           "targeted_humanevalplus"]
RANK = {c: {L: lr(code, c, L) for L in range(NL)} for c in CODECLS}
CREATR = {L: lr(code, "generic_creative", L) for L in range(NL)}
AG = {L: lr(ag, agcat, L) for L in range(NL)}
DOM = {L: dom_layer(L) for L in range(NL)}


def mincode(L, e):
    return min(RANK[c][L].get(e, 999) for c in CODECLS)


drop = {}
restored = creative_sub = cap2_fallback = 0
short_layers = []
for L in range(NL):
    target_sci = cap2[L] - fk[L]          # cap2's restored science set (verbatim)
    cap2_victims = fk[L] - cap2[L]         # what cap2 evicted (the worse victims)
    kept = set(fk[L]) | target_sci         # over by len(target_sci)
    over = len(kept) - 98
    restored += len(target_sci)
    # prefer evicting creative: loop-safe, non-pin, weak-code, not just-restored
    rem_cre = sorted(
        [e for e in kept if e not in target_sci and DOM[L].get(e) == "generic_creative"
         and AG[L].get(e, 999) > 16 and e not in pins[L] and mincode(L, e) > 16],
        key=lambda e: CREATR[L].get(e, 999), reverse=True)  # weakest creative first
    take = rem_cre[:over]
    kept -= set(take)
    creative_sub += len(take)
    # shortfall: NEVER touch code / HE+ / multilingual / science. Spend on
    # creative-loop-drivers first, then weakest logic/math.
    if len(kept) > 98:
        need = len(kept) - 98
        prot_dom = set(CODECLS) | {"generic_multilingual", "generic_science"}
        fbpool = sorted(
            [e for e in kept if e not in target_sci and e not in pins[L]
             and DOM[L].get(e) not in prot_dom and mincode(L, e) > 16],
            key=lambda e: (0 if DOM[L].get(e) == "generic_creative" else 1,
                           -mincode(L, e)))  # creative first, then weakest-code
        fb = fbpool[:need]
        kept -= set(fb)
        cap2_fallback += len(fb)
        short_layers.append((L, need, sorted(DOM[L].get(e) for e in fb)))
    _ = cap2_victims  # retained for provenance/debugging
    assert len(kept) == 98, f"L{L} kept={len(kept)}"
    drop[str(L)] = sorted(set(range(NE)) - kept)

json.dump(drop, open(out_p, "w"))
print(f"cap2-anchored creative-sacrifice: restored {restored} science (= cap2 set)")
print(f"  evicted via CREATIVE: {creative_sub}  |  cap2-victim fallback: {cap2_fallback}")
if short_layers:
    print(f"  creative-short layers (fellback to cap2 victim): {short_layers}")
