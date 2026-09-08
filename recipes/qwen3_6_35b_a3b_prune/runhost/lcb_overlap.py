#!/usr/bin/env python
"""Solve-set overlap between LCB columns: does armE DOMINATE armD or just reshuffle?

A +3/77 score gap is 3 problems. At n=77 that is not separable on the score axis alone,
so the question that matters is structural: are armE's solves a SUPERSET of armD's
(strict gain), or did it lose some and win more (a reshuffle, i.e. noise)?
"""
import json
import os
import sys

RES = "/srv/ml/eval_results/ream_arms/lcb_v6_77q"


def solves(name):
    jl = f"{RES}/{name}/lcb_result.samples.jsonl"
    out = {}
    for line in open(jl):
        try:
            r = json.loads(line)
        except Exception:
            continue
        k = r.get("doc_id", r.get("question_id", len(out)))
        p = None
        for f in ("pass", "passed", "correct", "pass@1", "graded", "is_correct"):
            if f in r:
                v = r[f]
                p = v if isinstance(v, bool) else (v >= 0.5)
                break
        out[k] = p
    return out


a, b = sys.argv[1], sys.argv[2]
A, B = solves(a), solves(b)
keys = sorted(set(A) & set(B), key=str)
sa = {k for k in keys if A[k]}
sb = {k for k in keys if B[k]}
print(f"{a}: {len(sa)}/{len(keys)}   {b}: {len(sb)}/{len(keys)}")
print(f"  both      = {len(sa & sb)}")
print(f"  only {a[:22]:22s} = {len(sa - sb)}")
print(f"  only {b[:22]:22s} = {len(sb - sa)}")
print(f"  neither   = {len(keys) - len(sa | sb)}")
if sa - sb or sb - sa:
    print("  -> RESHUFFLE (both directions moved)" if (sa - sb and sb - sa)
          else "  -> STRICT SUPERSET (one direction only)")
