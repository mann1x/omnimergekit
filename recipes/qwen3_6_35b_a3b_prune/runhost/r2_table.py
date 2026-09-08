#!/usr/bin/env python
"""R2 table with provenance + sample health, every column that exists on disk.

Score is read from summary.json .score ONLY -- omk_eval already picked the right
metric/filter per bench. Sampler is printed per cell because a greedy row and a sampled
row must never be tabulated together.

Sample health exists because a low pass@1 has two causes: the model got worse, or the
SCORER died on fences/empties/truncation (bug-015). A collapse must be shown to be the
former.
"""
import json
import os

RES = "/srv/ml/eval_results/ream_arms"
COLS = [
    ("base256e_imat", "base 256e (untuned anchor)"),
    ("pub184e_imat", "published 184e"),
    ("reamD_ourssal_nomerge", "armD  ours saliency, NO merge"),
    ("reamD_rpt", "armD  repeat (same GGUF)"),
    ("reamC_ourssal_merge", "armC  ours saliency + REAM merge"),
    ("reamB_reapsal_merge", "armB  REAP saliency + REAM merge  = STOCK REAM"),
    ("reamE_reapsal_nomerge", "armE  REAP saliency, NO merge"),
    ("reamF_rnorm_nomerge", "armF  rnorm cut, NO merge"),
    ("reamC_noimat_ctrl", "armC  no-imatrix control"),
]
BENCH = [("multipl_e_100", "MPE-100"), ("humaneval_full_think", "HE+")]


def cell(name, tmpl):
    p = f"{RES}/{tmpl}/{name}/summary.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d


print(f"{'column':46s} {'MPE-100':>9s} {'HE+':>9s}   sampler")
print("-" * 92)
for name, label in COLS:
    vals, samp = [], set()
    for tmpl, _ in BENCH:
        d = cell(name, tmpl)
        if d is None:
            vals.append("    --   ")
        else:
            vals.append(f"{d.get('score'):9.4f}")
            samp.add((d.get("sampler") or {}).get("name"))
    s = ",".join(sorted(x for x in samp if x)) or "-"
    print(f"{label:46s} {vals[0]} {vals[1]}   {s}")

print("\nsample health, multipl_e_100 (real model quality vs dead scorer)")
print(f"{'column':46s} {'n':>5s} {'empty':>6s} {'fenced':>7s} {'<5ch':>5s} {'p50ch':>6s}")
print("-" * 82)
for name, label in COLS:
    p = f"{RES}/multipl_e_100/{name}/mpe_result.samples.jsonl"
    if not os.path.exists(p):
        continue
    tot = empty = fenced = tiny = 0
    lens = []
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        tot += 1
        g = r.get("resps") or r.get("filtered_resps") or []
        while isinstance(g, list) and g:
            g = g[0]
        txt = g if isinstance(g, str) else ""
        lens.append(len(txt))
        if not txt.strip():
            empty += 1
        if "```" in txt:
            fenced += 1
        if len(txt.strip()) < 5:
            tiny += 1
    lens.sort()
    p50 = lens[len(lens) // 2] if lens else 0
    print(f"{label:46s} {tot:5d} {empty:6d} {fenced:7d} {tiny:5d} {p50:6d}")
