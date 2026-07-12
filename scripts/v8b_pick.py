#!/usr/bin/env python3
"""v8b_pick.py — diff candidate drop-maps against the fkbroad baseline and
classify the swap, so the v8b winner can be picked with ZERO GPU.

For each candidate (and any reference keep-set, e.g. v8b-safe) we compute,
per layer:
  keep = {0..127} - dropped
  swapped_IN  = keep_cand - keep_fkbroad   (experts v8b restores)
  swapped_OUT = keep_fkbroad - keep_cand   (experts v8b evicts)
then classify every swapped expert with the SAME competence data fkbroad used:
  * dominant class  = argmax_c wnorm rank (science_headroom's `dom`)
  * agentic t106 rank (loop-driver test): rank > 16 => LOOP-SAFE to restore
  * generic_code / targeted_lcb_medium_55 rank: <= STRONG_K => load-bearing code

Readout per candidate:
  sci_in       loop-safe science/ml restored  (dom in {science,multilingual} & ag>16)
  ml_in        of those, multilingual
  risky_in     science/ml restored that are LOOP-DRIVERS (ag<=16) -- MUST be ~0
  code_out     STRONG generic_code OR LCB experts evicted          -- MUST be ~0
  pins_freed   force-keep pins that actually dropped (real freed slot)
Winner = max sci_in at code_out~0 and risky_in==0.

Usage:
  v8b_pick.py <code_map.json> <agentic_t106_map.json> <fkbroad_drop.json> \
              <pins_str> <STRONG_K> <label=cand_drop.json> [<label=...> ...]
"""
import collections
import json
import sys

code_p, ag_p, fk_p, pins_s, strong_k = sys.argv[1:6]
STRONG_K = int(strong_k)
NL = 30
NE = 128
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
    """expert id -> rank (1 = strongest by wnorm) for class `cat` at layer L."""
    row = cats[cat][str(L)]
    order = sorted(row, key=lambda e: float(e["wnorm"]), reverse=True)
    return {e["id"]: i + 1 for i, e in enumerate(order)}


def dom_layer(L):
    """expert id -> dominant class (argmax wnorm across classes) at layer L."""
    best = {}
    for c in allc:
        for e in code[c][str(L)]:
            w = float(e["wnorm"])
            eid = e["id"]
            if eid not in best or w > best[eid][1]:
                best[eid] = (c, w)
    return {eid: cv[0] for eid, cv in best.items()}


def load_keep(p):
    """drop-map {'L':[dropped...]} OR keepmeta {'keep':{'L':[kept...]}} -> {L: set(kept)}."""
    d = json.load(open(p))
    if isinstance(d.get("keep"), dict):  # keepmeta (e.g. v8b_safe_keepmeta.json)
        return {int(L): set(int(x) for x in v) for L, v in d["keep"].items()}
    keep = {}
    for L in range(NL):
        dropped = set(int(x) for x in d[str(L)])
        keep[L] = set(range(NE)) - dropped
    return keep


pins = collections.defaultdict(set)
for tok in pins_s.split(","):
    L, e = tok.split(":")
    pins[int(L)].add(int(e))

fk_keep = load_keep(fk_p)
DOM = {L: dom_layer(L) for L in range(NL)}
AG = {L: lr(ag, agcat, L) for L in range(NL)}
CR = {L: lr(code, "generic_code", L) for L in range(NL)}
LR = {L: lr(code, "targeted_lcb_medium_55", L) for L in range(NL)}

print(f"STRONG_K={STRONG_K}  (a swapped-out expert is code_out if "
      f"generic_code rank<=K OR LCB rank<=K)")
print(f"{'label':<26} {'swap':>5} {'sci_in':>7} {'ml_in':>6} {'risky_in':>9} "
      f"{'code_out':>9} {'pins_freed':>11}")
print("-" * 82)

for arg in sys.argv[6:]:
    label, path = arg.split("=", 1)
    try:
        ck = load_keep(path)
    except Exception as e:  # noqa: BLE001
        print(f"{label:<26} LOAD-FAIL {path} ({e})")
        continue
    swap = sci_in = ml_in = risky_in = code_out = pins_freed = 0
    out_detail = []
    for L in range(NL):
        s_in = ck[L] - fk_keep[L]
        s_out = fk_keep[L] - ck[L]
        swap += len(s_in)
        for e in s_in:
            d = DOM[L].get(e)
            agr = AG[L].get(e, 999)
            if d in SCI:
                if agr > 16:
                    sci_in += 1
                    if d == "generic_multilingual":
                        ml_in += 1
                else:
                    risky_in += 1  # loop-driver science restored -> danger
        for e in s_out:
            cr = CR[L].get(e, 999)
            lcr = LR[L].get(e, 999)
            if cr <= STRONG_K or lcr <= STRONG_K:
                code_out += 1
                out_detail.append((L, e, cr, lcr))
        for e in pins[L]:
            if e not in ck[L]:
                pins_freed += 1
    print(f"{label:<26} {swap:>5} {sci_in:>7} {ml_in:>6} {risky_in:>9} "
          f"{code_out:>9} {pins_freed:>11}")
    if out_detail:
        od = ", ".join(f"{L}:{e}(c{cr}/l{lcr})" for (L, e, cr, lcr) in sorted(out_detail)[:12])
        print(f"    code_out detail: {od}{' ...' if len(out_detail) > 12 else ''}")

print("\nWinner rule: max sci_in with risky_in==0 and code_out~0 (no STRONG code/LCB lost).")
