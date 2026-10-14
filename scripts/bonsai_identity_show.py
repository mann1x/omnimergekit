#!/usr/bin/env python3
"""Print probe answers side by side per question:  bonsai_identity_show.py probe.jsonl [chars]"""
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
width = int(sys.argv[2]) if len(sys.argv) > 2 else 260
qs = list(dict.fromkeys(r["q"] for r in rows))
for q in qs:
    print("\n### " + q[:100])
    for tag in sorted({r["tag"] for r in rows}, reverse=True):
        for r in rows:
            if r["q"] == q and r["tag"] == tag:
                ans = r["answer"].split("</think>")[-1].strip().replace("\n", " ")
                print(f"  [{tag} s{r['seed']}] {ans[:width]}")
