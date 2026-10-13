#!/usr/bin/env python3
"""compare_drop_maps.py — diff two expert drop maps, per layer and in aggregate.

Built to answer a specific question: does rebuilding the competence map on a properly
terminated Tier-A and an AGENTIC, multi-language Tier-B change which experts the prune
evicts? "Even 1 expert different is worth trying" — so this reports the exact count, the
per-layer breakdown, and which experts moved in each direction, not just a similarity
score.

Also reports the overlap against CHANCE. Two independent 30-of-128 selections share
30*30/128 = 7.0 experts per layer by coincidence, so an unchanged-looking map is only
meaningful relative to that floor.
"""
from __future__ import annotations

import argparse
import json


def load(path):
    d = json.load(open(path))
    if isinstance(d, dict) and "drop" in d and isinstance(d["drop"], dict):
        d = d["drop"]
    return {str(k): set(v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference (e.g. the shipped v7-coder map)")
    ap.add_argument("--b", required=True, help="candidate (the new map)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--show-layers", type=int, default=30)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    layers = sorted(set(A) & set(B), key=lambda x: int(x) if str(x).isdigit() else x)
    only_a = set(A) ^ set(B)
    if only_a:
        print("WARNING: layer sets differ: %s" % sorted(only_a)[:10])

    tot_a = tot_b = same = 0
    rows = []
    for l in layers:
        a, b = A[l], B[l]
        tot_a += len(a); tot_b += len(b); same += len(a & b)
        rows.append((l, len(a), len(b), len(a & b), sorted(a - b), sorted(b - a)))

    n = len(layers)
    k_a = tot_a / n
    k_b = tot_b / n
    chance = k_a * k_b / args.experts
    print("%s: %d layers, %.1f dropped/layer   |   %s: %.1f dropped/layer   | experts=%d"
          % (args.label_a, n, k_a, args.label_b, k_b, args.experts))
    print("identical drops: %d of %d  (%.1f%%)   chance overlap %.1f/layer = %.1f%%"
          % (same, tot_a, 100.0 * same / max(tot_a, 1), chance, 100.0 * chance / max(k_a, 1)))
    changed = tot_a - same
    print("**%d experts differ** (%.1f per layer) — %s evicts them, %s does not"
          % (changed, changed / n, args.label_a, args.label_b))

    print("\n%-6s %-5s %-5s %-6s %-30s %s"
          % ("layer", "|A|", "|B|", "same", "only in %s (rescued)" % args.label_a,
             "only in %s (newly dropped)" % args.label_b))
    for l, na, nb, ns, oa, ob in rows[:args.show_layers]:
        print("%-6s %-5d %-5d %-6d %-30s %s"
              % (l, na, nb, ns, ",".join(map(str, oa))[:30], ",".join(map(str, ob))[:30]))

    if changed == 0:
        print("\nVERDICT: the two maps evict the SAME experts everywhere. Rebuilding the "
              "competence map changed nothing — the prune is insensitive to this axis.")
    else:
        print("\nVERDICT: %d differences. Worth building and evaluating." % changed)


if __name__ == "__main__":
    main()
