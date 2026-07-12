#!/usr/bin/env python3
"""gpqa_score_subset.py — score GPQA samples on the G_gap subset (and full).

Usage:
  gpqa_score_subset.py <gap.json> <label=glob_or_path> [<label=...> ...]

For each label, resolves the newest matching samples_*.jsonl, builds per-doc
flexible-extract correctness, and prints G_gap-subset + full-198 scores. This
is the per-rung readout for the v7->v8 recipe dissection.
"""
import glob
import json
import sys

gap = json.load(open(sys.argv[1]))
gap_ids = gap["gap_doc_ids"]
print(f"G_gap = {len(gap_ids)} doc_ids; v7_ref={gap.get('v7_score'):.4f} v8_ref={gap.get('v8_score'):.4f}")
print(f"{'rung':<26} {'G_gap':>8} {'full-198':>14}")
print("-" * 52)

for arg in sys.argv[2:]:
    label, patt = arg.split("=", 1)
    files = sorted(glob.glob(patt))
    if not files:
        print(f"{label:<26} {'NO SAMPLES':>8}  ({patt})")
        continue
    f = files[-1]
    corr = {}
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("filter") != "flexible-extract":
                continue
            corr[r["doc_id"]] = (r.get("exact_match") == 1.0)
    n = len(corr)
    full = sum(corr.values())
    gp = sum(1 for d in gap_ids if corr.get(d))
    print(f"{label:<26} {gp:>3}/{len(gap_ids):<4} {full:>5}/{n}={full/n:.4f}")
