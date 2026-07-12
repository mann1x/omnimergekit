#!/usr/bin/env python3
"""Validate a T223 force-keep drop map against intent.

Usage: validate_fk_map.py <base_map.json> <new_map.json> <t223_pins> <broad_pins> <label>
  t223_pins / broad_pins: "L:e,L:e,..." (may be empty string for the repro gate)

Checks, all of which must hold for PASS:
  - budget: len(dropped[L]) == EXPECTED (mode of base) for every layer
  - every T223 pin is KEPT (not in dropped[L]) in the new map
  - every broad pin is still KEPT
  - diff base->new: newly-KEPT == exactly the T223 pins that were dropped in base
                    newly-DROPPED (compensating evictions) count == #forced pins
  - no unexpected newly-kept (would mean generator drift, not a clean pin overlay)

Exit 0 = map matches intent, 1 = does NOT.
"""
import json
import sys


def load_map(p):
    d = json.load(open(p))
    return {int(k): set(int(x) for x in v)
            for k, v in d.items() if str(k).lstrip("-").isdigit()}


def parse_pins(s):
    out = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        ls, _, es = tok.partition(":")
        out.append((int(ls), int(es)))
    return out


def main():
    base_p, new_p, t223_s, broad_s, label = sys.argv[1:6]
    base = load_map(base_p)
    new = load_map(new_p)
    t223 = parse_pins(t223_s)
    broad = parse_pins(broad_s)

    # expected dropped/layer = mode of base layer sizes (98e -> 30)
    sizes = [len(base[ly]) for ly in base]
    exp = max(set(sizes), key=sizes.count)

    ok = True
    print(f"=== validate {label}: {new_p}")
    print(f"    layers={len(new)} expected_dropped/layer={exp} "
          f"t223_pins={len(t223)} broad_pins={len(broad)}")

    bad_budget = [ly for ly in new if len(new[ly]) != exp]
    if bad_budget:
        ok = False
        print("  [X] BUDGET broken: " +
              ", ".join(f"L{ly}={len(new[ly])}" for ly in bad_budget))
    else:
        print(f"  [OK] budget: all {len(new)} layers drop exactly {exp}")

    notkept = [(ly, e) for (ly, e) in t223 if e in new.get(ly, set())]
    already = [(ly, e) for (ly, e) in t223 if e not in base.get(ly, set())]
    if notkept:
        ok = False
        print("  [X] T223 pins STILL DROPPED: " +
              ", ".join(f"L{ly}e{e}" for ly, e in notkept))
    else:
        print(f"  [OK] all {len(t223)} T223 pins KEPT")
    if already:
        print("  [!] T223 pins already-kept in base (force-keep no-op): " +
              ", ".join(f"L{ly}e{e}" for ly, e in already))

    broad_drop = [(ly, e) for (ly, e) in broad if e in new.get(ly, set())]
    if broad_drop:
        ok = False
        print("  [X] BROAD pins DROPPED (should stay kept): " +
              ", ".join(f"L{ly}e{e}" for ly, e in broad_drop))
    else:
        print(f"  [OK] all {len(broad)} broad pins still kept")

    newly_kept = {ly: sorted(base[ly] - new[ly]) for ly in new if base.get(ly, set()) - new[ly]}
    newly_drop = {ly: sorted(new[ly] - base.get(ly, set())) for ly in new if new[ly] - base.get(ly, set())}
    if newly_kept:
        print("  newly KEPT (base-dropped -> now kept):")
        for ly in sorted(newly_kept):
            print(f"      L{ly}: {newly_kept[ly]}")
    if newly_drop:
        print("  newly DROPPED (compensating evictions):")
        for ly in sorted(newly_drop):
            print(f"      L{ly}: {newly_drop[ly]}")

    t223_set = {(ly, e) for (ly, e) in t223}
    nk_set = {(ly, e) for ly in newly_kept for e in newly_kept[ly]}
    forced = {(ly, e) for (ly, e) in t223 if e in base.get(ly, set())}
    unexpected = nk_set - t223_set
    missing = forced - nk_set
    if unexpected:
        ok = False
        print("  [X] UNEXPECTED newly-kept (generator drift?): " +
              ", ".join(f"L{ly}e{e}" for ly, e in sorted(unexpected)))
    if missing:
        ok = False
        print("  [X] MISSING force-keep: " +
              ", ".join(f"L{ly}e{e}" for ly, e in sorted(missing)))
    if not unexpected and not missing:
        print(f"  [OK] newly-kept set == forced T223 pins exactly ({len(nk_set)})")

    n_evict = sum(len(v) for v in newly_drop.values())
    if n_evict != len(forced):
        ok = False
        print(f"  [X] eviction count {n_evict} != forced pin count {len(forced)}")
    else:
        print(f"  [OK] evictions ({n_evict}) == forced pins ({len(forced)})")

    print(f"=== {label}: {'PASS (map matches intent)' if ok else 'FAIL (map != intent)'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
