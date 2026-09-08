#!/usr/bin/env python
"""Hybrid dose row, read from summary.json on disk (never the chain's SCORE log line)."""
import json
import os

R = "/srv/ml/eval_results/ream_arms"
ROWS = [
    ("armD  p=0  (ours)", "reamD_ourssal_nomerge"),
    ("armD_rpt  (replicate)", "reamD_rpt"),
    ("pub184e   (published)", "pub184e_imat"),
    ("armI  p=12 hybrid", "hybrid_p12_ourssal_reapfloor"),
    ("armJ  p=24 hybrid", "hybrid_p24_ourssal_reapfloor"),
    ("armE  REAP-select", "reamE_reapsal_nomerge"),
    ("armF  rnorm", "reamF_rnorm_nomerge"),
]

for t in ("multipl_e_100", "humaneval_full_think"):
    print("=" * 74)
    print(t)
    for lab, n in ROWS:
        p = os.path.join(R, t, n, "summary.json")
        if not os.path.exists(p):
            print("  %-24s (pending)" % lab)
            continue
        s = json.load(open(p))
        ts = s.get("token_stats") or {}
        c = ts.get("completion_tokens") or {}
        fr = ts.get("finish_reasons") or {}
        sam = (s.get("sampler") or {}).get("name")
        print("  %-24s score=%.4f  n=%s  p50=%s max=%s  fr=%s  sampler=%s"
              % (lab, s.get("score"), ts.get("n"), c.get("p50"), c.get("max"),
                 dict(fr), sam))
