#!/usr/bin/env python3
"""LCB budget-saturation + saturated-and-failed PASS/FAIL stats per cohort,
from lcb_result.samples.jsonl (per-problem passed + completion_tokens)."""
import json, glob, os, statistics as st

SAT = 12000  # near the thinking_token_budget=12288 cap (v6 convention)
COHORTS = [
    ("128e",      "/srv/ml/eval_results_128e_bs2",          ["lcb_medium_55","lcb_medium_55_v4"], ["lcb_medium_100","lcb_medium_100_v4"]),
    ("v6-coder",  "/srv/ml/eval_results_v6coder_bs2built",  ["lcb_medium_55_v4"], ["lcb_medium_100_v4"]),
    ("v7-coder",  "/srv/ml/eval_results_v7coder_g15f2440",  ["lcb_medium_55_v4"], ["lcb_medium_100_v4"]),
    ("v7-coderx", "/srv/ml/eval_results_v7coder_fs2440",    ["lcb_medium_55_v4"], ["lcb_medium_100_v4"]),
]

def find_samples(root, dirnames):
    for dn in dirnames:
        for sj in glob.glob(os.path.join(root, dn, "*", "lcb_result.samples.jsonl")):
            if ".bak" in sj: continue
            return sj
    return None

def analyze(path):
    n=0; sat=0; sat_fail=0
    pass_tok=[]; fail_tok=[]
    for line in open(path):
        line=line.strip()
        if not line: continue
        d=json.loads(line)
        ct=d.get("completion_tokens") or 0
        p=bool(d.get("passed"))
        n+=1
        (pass_tok if p else fail_tok).append(ct)
        if ct>=SAT:
            sat+=1
            if not p: sat_fail+=1
    return {
        "n": n, "sat": sat, "sat_fail": sat_fail,
        "mean_pass": round(st.mean(pass_tok)) if pass_tok else None,
        "mean_fail": round(st.mean(fail_tok)) if fail_tok else None,
        "n_pass": len(pass_tok), "n_fail": len(fail_tok),
    }

out={}
for clabel, root, d55, d100 in COHORTS:
    out[clabel]={}
    for key, dirs in (("lcb55",d55),("lcb100",d100)):
        p=find_samples(root, dirs)
        out[clabel][key]= (analyze(p) | {"src": p.replace("/srv/ml/","")}) if p else None
print(json.dumps(out, indent=1))
