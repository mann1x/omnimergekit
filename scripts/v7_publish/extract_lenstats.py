#!/usr/bin/env python3
"""Extract canonical-bench length stats (chars + tokens p50/p90/max) + score
for the 4 cohorts 128e / v6-coder / v7-coder / v7-coderx, from omk_eval
summary.json token_stats. One served-dir per bench (auto-discovered)."""
import json, glob, os, sys

COHORTS = [
    ("128e",      "/srv/ml/eval_results_128e_bs2"),
    ("v6-coder",  "/srv/ml/eval_results_v6coder_bs2built"),
    ("v7-coder",  "/srv/ml/eval_results_v7coder_g15f2440"),
    ("v7-coderx", "/srv/ml/eval_results_v7coder_fs2440"),
]
# canonical bench -> candidate dir names (128e drops the _v4 suffix on lcb)
BENCHES = [
    ("GPQA Diamond (198)",   ["gpqa_diamond_full"]),
    ("AIME 2024 (30)",       ["aime_30"]),
    ("LCB-medium-55",        ["lcb_medium_55_v4","lcb_medium_55"]),
    ("LCB-medium-100",       ["lcb_medium_100_v4","lcb_medium_100"]),
    ("MultiPL-E-100 (300)",  ["multipl_e_100"]),
    ("MATH-500 (100)",       ["math500_100"]),
    ("GSM8K (100)",          ["gsm8k_100"]),
    ("IFEval (100)",         ["ifeval_100"]),
    ("HumanEval (164)",      ["humaneval_full"]),
    ("HumanEval+ (164)",     ["humanevalplus_full"]),
    ("ARC-Challenge (1172)", ["arc_challenge_full"]),
]

def load(root, dirnames):
    for dn in dirnames:
        base = os.path.join(root, dn)
        if not os.path.isdir(base):
            continue
        cands = sorted(glob.glob(os.path.join(base, "*", "summary.json")))
        # skip *.bak celldirs
        cands = [c for c in cands if ".bak" not in c]
        if not cands:
            continue
        return json.load(open(cands[0])), cands[0]
    return None, None

def trip(d, key):
    s = (d or {}).get(key) or {}
    return s.get("p50"), s.get("p90"), s.get("max")

out = {"chars": {}, "tokens": {}, "score": {}, "src": {}}
for blabel, dirs in BENCHES:
    out["chars"][blabel] = {}
    out["tokens"][blabel] = {}
    out["score"][blabel] = {}
    out["src"][blabel] = {}
    for clabel, root in COHORTS:
        d, src = load(root, dirs)
        if d is None:
            out["chars"][blabel][clabel] = None
            out["tokens"][blabel][clabel] = None
            out["score"][blabel][clabel] = None
            out["src"][blabel][clabel] = None
            continue
        ts = d.get("token_stats") or {}
        out["chars"][blabel][clabel]  = trip(ts, "completion_chars")
        out["tokens"][blabel][clabel] = trip(ts, "completion_tokens")
        out["score"][blabel][clabel]  = d.get("score")
        out["src"][blabel][clabel]    = src.replace("/srv/ml/","")

print(json.dumps(out, indent=1))
