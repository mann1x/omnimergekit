#!/usr/bin/env python3
"""What fields do the SHIPPED Qwen3.6 competence maps actually carry?

Answering the HF question about rnorm-vs-frequency correlation requires knowing whether
rnorm/wnorm were populated at all. STATE.md says the maps were built `--tc-only`; verify
that against the artifacts rather than trusting the note.
"""
import collections
import json
import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
for f in ("competence_qwen35b.json", "competence_qwen35b_coder.json",
          "competence_qwen35b_coder_lcbmpe.json", "competence_qwen35b_coder_lcbmpeife.json"):
    p = os.path.join(D, f)
    if not os.path.exists(p):
        print(f, "MISSING")
        continue
    d = json.load(open(p))
    cats = d.get("categories", {})
    keys = collections.Counter()
    nonzero = collections.Counter()
    n = 0
    for cat, layers in cats.items():
        for li, experts in layers.items():
            for e in experts:
                n += 1
                for k, v in e.items():
                    keys[k] += 1
                    if isinstance(v, (int, float)) and v:
                        nonzero[k] += 1
    print(f"\n{f}")
    print("  top-level keys :", sorted(d))
    print("  categories     :", sorted(cats))
    print("  layers         :", len(next(iter(cats.values()))) if cats else 0)
    print("  expert cells   :", n)
    print("  per-expert keys:", dict(keys))
    print("  NON-ZERO       :", dict(nonzero))
    for k in ("meta", "args", "gen"):
        if k in d:
            print(f"  {k}:", json.dumps(d[k])[:400])
