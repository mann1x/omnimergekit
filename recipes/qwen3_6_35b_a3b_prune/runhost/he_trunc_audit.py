#!/usr/bin/env python
"""Is HE+ 'len-term 0/164' a measured zero or a missing field?

fr.get('length', 0) collapses two different states into the same printed 0:
  (a) the server terminated no completion on length  -> a real measurement
  (b) token_stats has no finish_reasons at all       -> nothing was measured
The OMK_CAP_CHECK sentinel reports capped=36/164 on one HE+ column, so at least
one of these columns is NOT a clean zero. Print the raw structure and decide.
"""
import glob
import json
import os

R = "/srv/ml/eval_results/ream_arms"

for t in ("humaneval_plus_full_think", "humaneval_full_think", "multipl_e_100"):
    paths = sorted(glob.glob(f"{R}/{t}/*/summary.json"))
    if not paths:
        continue
    print(f"===== {t}  ({len(paths)} columns)")
    for p in paths:
        name = os.path.basename(os.path.dirname(p))
        s = json.load(open(p))
        ts = s.get("token_stats")
        if ts is None:
            print(f"  {name:44s} score={s.get('score')}  token_stats=ABSENT")
            continue
        has_fr = "finish_reasons" in ts
        fr = ts.get("finish_reasons")
        c = ts.get("completion_tokens") or {}
        print(f"  {name:44s} score={s.get('score')}  n={ts.get('n')}  "
              f"has_finish_reasons={has_fr}  fr={fr}  "
              f"p50={c.get('p50')} max={c.get('max')}")
        for k in ("cap_check", "capped", "omk_cap_check"):
            if k in s:
                print(f"      summary[{k}] = {json.dumps(s[k])[:300]}")
