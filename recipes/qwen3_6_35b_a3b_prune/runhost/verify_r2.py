#!/usr/bin/env python
"""Verify the R2 table before anyone reads a number off it.

Two independent checks, because a low score has two very different causes:
  * provenance -- every cell must be the SAME basis (greedy/template_default) or the
    columns are not comparable at all;
  * sample health -- lm-eval reports a technically-correct pass@1 of ~0 when the SCORER
    dies on markdown fences / empty generations / truncation (bug-015). A collapse must
    be shown to be the model producing worse code, not the harness failing to grade it.
"""
import json
import os
import re
from collections import Counter

RES = "/srv/ml/eval_results/ream_arms"
COLS = ["reamD_ourssal_nomerge", "reamC_ourssal_merge", "base256e_imat",
        "pub184e_imat", "reamD_rpt", "reamC_noimat_ctrl"]
BENCH = ["multipl_e_100", "humaneval_full_think"]

print("=== provenance (score / metric / filter / sampler) ===")
for n in COLS:
    for t in BENCH:
        s = f"{RES}/{t}/{n}/summary.json"
        if not os.path.exists(s):
            print(f"  {n:24s} {t:22s} NO SUMMARY")
            continue
        d = json.load(open(s))
        sm = (d.get("sampler") or {})
        print(f"  {n:24s} {t:22s} score={d.get('score')} metric={d.get('metric')} "
              f"filter={d.get('filter')} sampler={sm.get('name')}")

print("\n=== sample health, multipl_e_100 (is a low score real, or a dead scorer?) ===")
for n in COLS:
    p = f"{RES}/multipl_e_100/{n}/mpe_result.samples.jsonl"
    if not os.path.exists(p):
        print(f"  {n:24s} no samples file")
        continue
    tot = empty = fenced = tiny = 0
    lens = []
    passed = 0
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        tot += 1
        g = r.get("resps") or r.get("filtered_resps") or []
        txt = ""
        while isinstance(g, list) and g:
            g = g[0]
        if isinstance(g, str):
            txt = g
        lens.append(len(txt))
        if not txt.strip():
            empty += 1
        if "```" in txt:
            fenced += 1
        if len(txt.strip()) < 5:
            tiny += 1
        for k in ("pass_at_1", "pass@1", "acc"):
            if isinstance(r.get(k), (int, float)):
                passed += r[k] > 0
                break
    lens.sort()
    p50 = lens[len(lens) // 2] if lens else 0
    print(f"  {n:24s} n={tot:4d} empty={empty:3d} fenced={fenced:4d} <5char={tiny:3d} "
          f"p50_chars={p50:5d} scored_pass={passed}")

print("\n=== why reamC_noimat_ctrl produced nothing ===")
log = f"{RES}/multipl_e_100/reamC_noimat_ctrl/server.log"
if os.path.exists(log):
    txt = open(log, errors="replace").read()
    hits = [ln for ln in txt.splitlines()
            if re.search(r"missing tensor|error|failed to load", ln, re.I)]
    for ln in hits[:5]:
        print("   ", ln[:150])
else:
    print("    no server.log at", log)
