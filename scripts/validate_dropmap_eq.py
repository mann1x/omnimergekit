#!/usr/bin/env python3
"""Set-wise per-layer equality check between two drop maps.

Usage: validate_dropmap_eq.py <a.json> <b.json>
Exit 0 + 'EQUAL' if every layer's dropped-expert SET matches (order/format
agnostic); exit 1 + the per-layer diff otherwise. Used to prove the
force-keep-capable generator reproduces C6v3lcb byte-for-set before trusting
its --force-keep output.
"""
import json
import sys


def dl(path):
    m = json.load(open(path))
    m = m.get("per_layer_drop", m.get("drop", m))
    return {str(k): set(v) for k, v in m.items() if str(k).lstrip("-").isdigit()}


a, b = dl(sys.argv[1]), dl(sys.argv[2])
keys = sorted(set(a) | set(b), key=int)
bad = 0
for L in keys:
    sa, sb = a.get(L, set()), b.get(L, set())
    if sa != sb:
        bad += 1
        print(f"  L{int(L):02d} DIFF  only_a={sorted(sa - sb)}  only_b={sorted(sb - sa)}")
if bad:
    print(f"NOT_EQUAL ({bad}/{len(keys)} layers differ)")
    sys.exit(1)
print(f"EQUAL ({len(keys)} layers, {sum(len(v) for v in a.values())} total drops)")
