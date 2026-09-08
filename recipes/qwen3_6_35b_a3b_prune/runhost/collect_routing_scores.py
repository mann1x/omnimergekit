#!/usr/bin/env python3
"""Tabulate every banked cell in the routing suite from summary.json ONLY.

summary.json .score is the canonical number -- omk_eval already picked the right
metric/filter per bench. Reading raw results_*.json is the 2026-05-23 GPQA-1.52% /
math500-41% false-alarm trap. Sampler is printed because a greedy cohort and a
sampled cohort must never share a table.
"""
import glob
import json

rows = []
for p in sorted(glob.glob("/srv/ml/eval_results_b604/*/*/summary.json")):
    try:
        d = json.load(open(p))
    except Exception as e:                                  # a broken cell is a finding
        print("ERR", p, e)
        continue
    parts = p.split("/")
    rows.append((parts[-3], parts[-2], d.get("score"), d.get("metric"),
                 (d.get("sampler") or {}).get("name"), d.get("verdict")))

hdr = "%-22s %-18s %8s  %-26s %-17s %s" % (
    "bench", "cell", "score", "metric", "sampler", "verdict")
print(hdr)
print("-" * len(hdr))
for b, c, s, m, sm, v in rows:
    print("%-22s %-18s %8s  %-26s %-17s %s" % (b, c, s, m, sm, v))
print("total cells:", len(rows))
