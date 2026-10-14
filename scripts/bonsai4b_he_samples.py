#!/usr/bin/env python3
"""Sanity table for lm-eval humaneval sample files: pass count, empty / junk / fenced / think-tag generations, lengths.

    bonsai4b_he_samples.py OUT_DIR [OUT_DIR ...]      # each is an lm-eval --output_path
"""
import glob
import json
import sys

for d in sys.argv[1:]:
    fs = sorted(glob.glob(f"{d}/*/samples_humaneval_*.jsonl"))
    if not fs:
        print(d, "no samples")
        continue
    rows = [json.loads(line) for line in open(fs[-1])]
    gen = [r["resps"][0][0] for r in rows]
    ln = sorted(len(x) for x in gen)
    ps = sum(r["pass@1"] for r in rows)
    print(f"{d.rstrip('/').split('/')[-1]}: n={len(rows)} pass={ps:.0f} ({100 * ps / len(rows):.2f}%) "
          f"empty={sum(1 for x in gen if not x.strip())} short<5={sum(1 for x in gen if len(x.strip()) < 5)} "
          f"fence={sum(1 for x in gen if '```' in x)} think={sum(1 for x in gen if '<think>' in x)} "
          f"len p10/p50/max={ln[len(ln) // 10]}/{ln[len(ln) // 2]}/{ln[-1]} over3000ch={sum(1 for x in gen if len(x) > 3000)}")
