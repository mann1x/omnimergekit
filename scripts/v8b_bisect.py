#!/usr/bin/env python3
"""Bisect the v8b-light swapped-IN experts: which restored experts are
terminator-competitive (agentic_eog loop-drivers, do NOT restore) vs which
are high-science/multilingual but NOT terminator-competitive (loop-safe to
restore). Decides whether a loop-safe partial science recovery on v8 is even
possible, purely from the competence maps (zero GPU).

Usage:
  v8b_bisect.py <code_map.json> <agentic_eog_map.json> <fkbroad_drop.json> \
                <v8b_light_drop.json> <fkbroad_summary.json> [--emit-map OUT.json]

"restored" = experts fkbroad DROPPED but v8b-light KEPT  (fk[L] - v8b[L]).
agentic_rank: rank of expert within its layer by agentic_eog wnorm (1=top =
most terminator-competitive). sci/ml dominance from the code map's argmax class.
"""
import collections
import json
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
code_p, agentic_p, fk_p, v8b_p, summ_p = args[:5]
emit_map = None
if "--emit-map" in sys.argv:
    emit_map = sys.argv[sys.argv.index("--emit-map") + 1]

NL = 30
SIG_SCI = {"generic_science", "generic_multilingual"}


def load_cats(path):
    d = json.load(open(path))
    cats = d.get("categories", d)
    return cats


def load_drop(path):
    m = json.load(open(path))
    it = m.items() if isinstance(m, dict) else enumerate(m)
    return {int(k): set(int(x) for x in v) for k, v in it}


def parse_pins(summary_path):
    d = json.load(open(summary_path))
    fkstr = None
    for k in ("force_keep", "force_keep_str", "pins", "force_keep_pins"):
        if k in d:
            fkstr = d[k]
            break
    if fkstr is None and isinstance(d.get("args"), dict):
        fkstr = d["args"].get("force_keep")  # fkbroad stores pins under args.force_keep
    if fkstr is None:
        return {}, "force_keep not found; keys=" + ",".join(sorted(d.keys()))
    pins = {}
    for tok in str(fkstr).replace(" ", "").split(","):
        if ":" in tok:
            L, e = tok.split(":")
            pins.setdefault(int(L), set()).add(int(e))
    return pins, f"parsed {sum(len(v) for v in pins.values())} pins"


code = load_cats(code_p)
ag = load_cats(agentic_p)
print("== code map categories:", list(code.keys()))
print("== agentic map categories:", list(ag.keys()))

# pick the agentic category
ag_cat = None
for c in ag:
    cl = c.lower()
    if "agentic" in cl or "eog" in cl or "terminat" in cl or "t106" in cl:
        ag_cat = c
        break
if ag_cat is None and len(ag) == 1:
    ag_cat = list(ag)[0]
print("== using agentic category:", ag_cat)
ALLC = list(code.keys())


def layer_rank(cats, cat, L):
    """id -> 1-based rank within layer L by wnorm desc."""
    row = cats[cat][str(L)]
    order = sorted(row, key=lambda e: float(e["wnorm"]), reverse=True)
    return {e["id"]: i + 1 for i, e in enumerate(order)}


def dom_class(L):
    best = {}
    for c in ALLC:
        for e in code[c][str(L)]:
            w = float(e["wnorm"])
            eid = e["id"]
            if eid not in best or w > best[eid][1]:
                best[eid] = (c, w)
    return {eid: cv[0] for eid, cv in best.items()}


fk = load_drop(fk_p)
v8b = load_drop(v8b_p)
pins, pin_msg = parse_pins(summ_p)
print("== pins:", pin_msg)

# per-layer agentic rank of the pins (calibrate the terminator neighborhood)
pin_ag_ranks = []
for L, es in pins.items():
    ar = layer_rank(ag, ag_cat, L)
    for e in es:
        if e in ar:
            pin_ag_ranks.append(ar[e])
if pin_ag_ranks:
    pin_ag_ranks.sort()
    print(f"== pin agentic_eog ranks: min={pin_ag_ranks[0]} "
          f"median={pin_ag_ranks[len(pin_ag_ranks)//2]} max={pin_ag_ranks[-1]} "
          f"(n={len(pin_ag_ranks)})")

# gather restored experts with their ranks
restored = []  # (L, e, agentic_rank, sci_rank, ml_rank, dom)
for L in range(NL):
    si = fk.get(L, set()) - v8b.get(L, set())
    if not si:
        continue
    ar = layer_rank(ag, ag_cat, L)
    sr = layer_rank(code, "generic_science", L) if "generic_science" in code else {}
    mr = layer_rank(code, "generic_multilingual", L) if "generic_multilingual" in code else {}
    dom = dom_class(L)
    for e in si:
        restored.append((L, e, ar.get(e, 999), sr.get(e, 999), mr.get(e, 999), dom.get(e, "?")))

print(f"\n== restored (fkbroad-dropped, v8b-kept) experts: {len(restored)}")

# agentic_rank histogram of restored experts
hist = collections.Counter()
for (_, _, agr, _, _, _) in restored:
    b = "1-8" if agr <= 8 else "9-16" if agr <= 16 else "17-30" if agr <= 30 else "31-60" if agr <= 60 else "61+"
    hist[b] += 1
print("agentic_rank histogram of restored:",
      {k: hist[k] for k in ("1-8", "9-16", "17-30", "31-60", "61+")})

domc = collections.Counter(r[5] for r in restored)
print("dominant-class of restored:", dict(domc.most_common()))

# classification sweep over terminator-neighborhood threshold T
print("\n== loop-driver vs safe-science separability sweep ==")
print(f"{'T(agtop)':>8} {'loopDriver':>11} {'safeSci(domSci&ag>T)':>22} {'safeSciAny(ag>T)':>17}")
for T in (8, 12, 16, 20, 24):
    loop_drivers = [r for r in restored if r[2] <= T]
    safe_sci = [r for r in restored if r[2] > T and r[5] in SIG_SCI]
    safe_any = [r for r in restored if r[2] > T]
    print(f"{T:>8} {len(loop_drivers):>11} {len(safe_sci):>22} {len(safe_any):>17}")

# correlation between science rank and agentic rank among restored sci/ml experts
sci_restored = [r for r in restored if r[5] in SIG_SCI]
print(f"\n== {len(sci_restored)} restored experts have DOMINANT class science/multilingual")
if sci_restored:
    # how terminator-competitive are the science experts?
    ag_of_sci = sorted(r[2] for r in sci_restored)
    n = len(ag_of_sci)
    print(f"   their agentic_eog ranks: min={ag_of_sci[0]} p25={ag_of_sci[n//4]} "
          f"median={ag_of_sci[n//2]} p75={ag_of_sci[3*n//4]} max={ag_of_sci[-1]}")
    in_term = sum(1 for r in sci_restored if r[2] <= 16)
    print(f"   science experts INSIDE terminator neighborhood (agentic_rank<=16): "
          f"{in_term}/{n} ({100*in_term/n:.0f}%)")

# VERDICT
T = 16
safe_sci = [r for r in restored if r[2] > T and r[5] in SIG_SCI]
print(f"\n== VERDICT (T={T}) ==")
print(f"   loop-safe science/ml restorable experts: {len(safe_sci)}")
if len(safe_sci) >= 20:
    print("   SEPARABLE — a loop-safe partial science recovery looks constructible.")
    print("   Top loop-safe science candidates (L, e, agRank, sciRank, dom):")
    for r in sorted(safe_sci, key=lambda x: x[3])[:25]:
        print(f"     L{r[0]:>2} e{r[1]:>3}  ag={r[2]:>3} sci={r[3]:>3} ml={r[4]:>3} dom={r[5]}")
else:
    print("   NOT SEPARABLE — high-science experts are terminator-competitive.")
    print("   Selection cannot recover science loop-free; training (KD+EOG) is the path.")

# optional: emit a candidate loop-safe drop map (restore only safe_sci; re-drop
# the equal number of lowest combined-value kept non-pin experts per layer)
if emit_map and len(safe_sci) >= 20:
    by_layer = collections.defaultdict(list)
    for r in safe_sci:
        by_layer[r[0]].append(r)
    new_drop = {L: set(v8b.get(L, set())) for L in range(NL)}  # start from v8b? no — from fkbroad
    new_drop = {L: set(fk.get(L, set())) for L in range(NL)}   # start from v8(fkbroad)=0/48
    sci_rank_full = {L: layer_rank(code, "generic_science", L) for L in range(NL)}
    code_rank_full = {L: layer_rank(code, "generic_code", L) for L in range(NL)}
    ag_rank_full = {L: layer_rank(ag, ag_cat, L) for L in range(NL)}
    for L, rs in by_layer.items():
        kept = set(range(128)) - new_drop[L]
        # candidates to re-drop: kept, NOT a pin, NOT terminator-competitive
        # (agentic_rank>16 so we never drop a terminator helper), lowest (sci+code) value.
        droppable = [e for e in kept
                     if e not in pins.get(L, set())
                     and ag_rank_full[L].get(e, 999) > 16]
        droppable.sort(key=lambda e: (sci_rank_full[L].get(e, 999) + code_rank_full[L].get(e, 999)),
                       reverse=True)  # worst (highest rank number) first
        for r in rs:
            e_in = r[1]
            if not droppable:
                break
            e_out = droppable.pop(0)
            new_drop[L].discard(e_in)  # restore safe-science expert
            new_drop[L].add(e_out)     # drop a low-value non-pin expert
    out = {str(L): sorted(new_drop[L]) for L in range(NL)}
    # sanity: 30 dropped per layer, all pins kept
    bad = [L for L in range(NL) if len(new_drop[L]) != len(fk.get(L, set()))]
    pin_viol = [(L, e) for L, es in pins.items() for e in es if e in new_drop[L]]
    json.dump(out, open(emit_map, "w"), indent=0)
    print(f"\n== emitted candidate map -> {emit_map}")
    print(f"   per-layer-drop-count violations: {bad}")
    print(f"   pin violations (pinned expert dropped): {pin_viol}")
