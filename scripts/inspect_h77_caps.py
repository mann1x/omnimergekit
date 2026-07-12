#!/usr/bin/env python3
# Classify lcb_hard_77 length-capped completions: genuine cut-off (more ctx helps)
# vs degenerate loop (more ctx just loops longer). Reads the sqlite response cache.
import os
import pickle
import sqlite3
from collections import Counter

BASE = "/srv/ml/eval_results_lcb_hard77/lcb_hard_77/%s/sqlite_cache/lcb_hard_77_%s.db"

def rep_frac(txt, n=12, tail=2500):
    """Max repetition fraction of n-word shingles in the last `tail` chars.
    High (>~0.15) => the tail is dominated by one repeated phrase (loop)."""
    t = txt[-tail:]
    w = t.split()
    if len(w) < n + 5:
        return 0.0
    sh = [tuple(w[i:i+n]) for i in range(len(w)-n)]
    c = Counter(sh)
    return c.most_common(1)[0][1] / max(1, len(sh))

def has_code(txt):
    return ("class Solution" in txt) or ("def " in txt)

for sn in ["std16-h77", "coderx-h77"]:
    db = BASE % (sn, sn)
    if not os.path.exists(db):
        print(f"=== {sn}: (no cache) ===")
        continue
    c = sqlite3.connect(db)
    rows = [pickle.loads(v) for (v,) in c.execute("select value from responses").fetchall()]
    caps = [d for d in rows if d.get("finish_reason") == "length"]
    print(f"\n========== {sn}: {len(caps)} length-capped / {len(rows)} done ==========")
    genuine = loops = 0
    for d in sorted(caps, key=lambda x: x.get("task_id", "")):
        # the full generated text spans reasoning + answer; concat the text fields
        full = ""
        for k in ("cleaned", "completion"):
            v = d.get(k)
            if isinstance(v, str):
                full += v
        rf = rep_frac(full)
        verdict = "LOOP" if rf >= 0.15 else "genuine-cut"
        if verdict == "LOOP":
            loops += 1
        else:
            genuine += 1
        tail = full[-180:].replace("\n", "\\n")
        print(f"  {d.get('task_id'):22} toks={str(d.get('completion_tokens')):>6} "
              f"code={int(has_code(full))} rep={rf:.3f} {verdict}")
        print(f"        tail: {tail}")
    print(f"  --> {sn}: genuine-cut={genuine}  LOOP={loops}")
