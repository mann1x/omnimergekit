#!/usr/bin/env python3
"""extend_v8b_safe.py — go beyond v8b-safe's 85 restored science the SAME way
v8b-safe did it (filtered greedy swap), NOT via generator re-weight (which the
sweep proved drags in loop-drivers + evicts strong code).

Two extension modes, both starting from v8b_safe_keepmeta:
  FREE  : evict weak NON-protected NON-science generalists -> add best dropped
          loop-safe science (agentic t106 rank>16). 1:1 swap, 98/layer preserved.
          Protected = pins OR rank<=16 in {code,lcb,he,hep} OR rank<=8 in {math,logic}.
          This is pure upside: no pin touched, no protected-code lost.
  UNPIN : additionally un-pin a given pin set (free those slots) and fill with
          the next-best loop-safe science. Costs whatever code the pin held.

Writes drop-maps so v8b_pick.py can score them vs fkbroad. Also prints the
per-layer FREE headroom so we know if un-pinning is even necessary.

Usage:
  extend_v8b_safe.py <keepmeta.json> <code_map.json> <agentic_t106.json> \
      <pins_str> <out_dir> [--unpin L:e,L:e,...] [--max-free-per-layer N]
"""
import collections
import json
import sys

keepmeta_p, code_p, ag_p, pins_s, out_dir = sys.argv[1:6]
unpin_s = ""
max_free = 99
av = sys.argv[6:]
i = 0
while i < len(av):
    if av[i] == "--unpin":
        unpin_s = av[i + 1]
        i += 2
    elif av[i] == "--max-free-per-layer":
        max_free = int(av[i + 1])
        i += 2
    else:
        i += 1

NL, NE = 30, 128
SCI = {"generic_science", "generic_multilingual"}
STRONG = {"generic_code": 16, "targeted_lcb_medium_55": 16,
          "targeted_humaneval": 16, "targeted_humanevalplus": 16,
          "generic_math": 8, "generic_logic": 8}


def load_cats(p):
    d = json.load(open(p))
    return d.get("categories", d)


code = load_cats(code_p)
ag = load_cats(ag_p)
agcat = next((c for c in ag if any(k in c.lower() for k in ("agentic", "eog", "t106"))),
             list(ag)[0] if len(ag) == 1 else None)
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


km = json.load(open(keepmeta_p))
keep = {int(L): set(int(x) for x in v) for L, v in km["keep"].items()}
for L in range(NL):
    assert len(keep[L]) == 98, f"layer {L} has {len(keep[L])} kept, expected 98"

pins = collections.defaultdict(set)
for tok in pins_s.split(","):
    L, e = tok.split(":")
    pins[int(L)].add(int(e))
unpin = collections.defaultdict(set)
if unpin_s:
    for tok in unpin_s.split(","):
        L, e = tok.split(":")
        unpin[int(L)].add(int(e))

DOM = {L: dom_layer(L) for L in range(NL)}
AG = {L: lr(ag, agcat, L) for L in range(NL)}
RANK = {cls: {L: lr(code, cls, L) for L in range(NL)} for cls in STRONG}
SCIR = {L: lr(code, "generic_science", L) for L in range(NL)}


def min_rank_all(L, e):
    return min(RANK[c][L].get(e, 999) for c in STRONG)


def build(do_free, unpin_map):
    """return (drop_map dict, stats)."""
    drop = {}
    n_free = n_unpin_fill = 0
    free_per_layer = {}
    for L in range(NL):
        kept = set(keep[L]) - unpin_map[L]            # un-pinned slots freed
        pinset = pins[L] - unpin_map[L]
        # protected kept experts (never evict)
        prot = set(pinset)
        for c, K in STRONG.items():
            prot |= {e for e in kept if RANK[c][L].get(e, 999) <= K}
        # dropped loop-safe science, best science first
        dropped = set(range(NE)) - kept - unpin_map[L]
        avail = sorted([e for e in dropped if DOM[L].get(e) in SCI and AG[L].get(e, 999) > 16],
                       key=lambda e: SCIR[L].get(e, 999))
        added = []
        # 1) fill the un-pinned slots first
        nfill = len(unpin_map[L])
        for e in avail[:nfill]:
            added.append(e)
        n_unpin_fill += len(added)
        ptr = len(added)
        # 2) optional free swaps: evict weakest non-protected non-science generalist
        nf = 0
        if do_free:
            evictable = sorted([e for e in kept if e not in prot and DOM[L].get(e) not in SCI],
                               key=lambda e: -min_rank_all(L, e))  # weakest first
            for ev in evictable:
                if ptr >= len(avail) or nf >= max_free:
                    break
                kept.discard(ev)
                added.append(avail[ptr])
                ptr += 1
                nf += 1
        free_per_layer[L] = nf
        n_free += nf
        kept |= set(added)
        assert len(kept) == 98, f"L{L} kept={len(kept)} after extend"
        drop[str(L)] = sorted(set(range(NE)) - kept)
    return drop, dict(free=n_free, unpin_fill=n_unpin_fill, free_per_layer=free_per_layer)


# E1: FREE extension only (no un-pin); cap = --max-free-per-layer
e1, s1 = build(True, collections.defaultdict(set))
json.dump(e1, open(f"{out_dir}/v8b_ext_free_cap{max_free}_drop_map.json", "w"))
print(f"E1 FREE extension (cap {max_free}/layer): +{s1['free']} clean science swaps "
      f"beyond v8b-safe (no pin touched, no protected code lost)")
print(f"   per-layer free swaps: {dict(sorted((k, v) for k, v in s1['free_per_layer'].items() if v))}")

# E0: un-pin trio + fill (only if --unpin given)
if unpin_s:
    e0, s0 = build(False, unpin)
    json.dump(e0, open(f"{out_dir}/v8b_ext_unpin_drop_map.json", "w"))
    print(f"E0 UNPIN({unpin_s}): filled {s0['unpin_fill']} freed slots with loop-safe science")
    # E2: un-pin trio AND free swaps
    e2, s2 = build(True, unpin)
    json.dump(e2, open(f"{out_dir}/v8b_ext_unpin_free_drop_map.json", "w"))
    print(f"E2 UNPIN+FREE: {s2['unpin_fill']} unpin-fill + {s2['free']} free swaps")
