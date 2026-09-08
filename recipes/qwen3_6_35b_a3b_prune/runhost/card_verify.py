"""Verify every number on the Qwen3.6-27B-A3B-Coder HF card against its banked
summary.json. Scores come from summary.json .score ONLY -- never raw results_*."""
import json, os, glob

ROOT = "/srv/ml/eval_results/qwen_suite"
# card column -> cell name
COLS = {"ours(t10)": "qwencodermpe_t10_q6k",
        "base256e":  "qwen256e_q6k",
        "coder(lcb)": "qwencoder_q6k"}
# card row -> bench dir + published triple (ours, base, coder)
ROWS = [
    ("GPQA-Diamond",  "gpqa_diamond_full",      (0.773, 0.833, 0.793)),
    ("MATH-500",      "math500_100",            (0.620, 0.730, 0.620)),
    ("AIME",          "aime_30",                (0.733, 0.633, 0.767)),
    ("LiveCodeBench", "lcb_v6_77q",             (0.688, 0.714, 0.688)),
    ("IFEval",        "ifeval_100",             (0.730, 0.960, 0.840)),
    ("HumanEval",     "humaneval_full_think",   (0.970, 0.970, 0.963)),
    ("GSM8K",         "gsm8k_100_boxed",        (0.970, 0.960, 0.980)),
    ("ARC-Challenge", "arc_challenge_full",     (0.944, 0.935, 0.933)),
    ("MultiPL-E",     "multipl_e_100",          (0.840, 0.827, 0.670)),
]

def banked(bench, cell):
    p = os.path.join(ROOT, bench, cell, "summary.json")
    if not os.path.exists(p):
        # gsm8k/humaneval also live in the non-boxed/non-think dir for some cells
        return None, "NO summary.json", None
    d = json.loads(open(p).read())
    s = d.get("score")
    samp = (d.get("sampler") or {}).get("name")
    return s, None, samp

print("%-15s %-22s %-11s %-9s %-9s %-7s %s" % (
    "row", "cell", "column", "card", "banked", "delta", "sampler"))
print("-" * 92)
bad = []
for row, bench, pub in ROWS:
    for (col, cell), p in zip(COLS.items(), pub):
        s, err, samp = banked(bench, cell)
        if s is None:
            print("%-15s %-22s %-11s %-9.3f %-9s %-7s %s" % (row, cell, col, p, "MISSING", "?", err))
            bad.append((row, col, "missing"))
            continue
        d = s - p
        flag = "" if abs(d) < 0.0051 else "  <== MISMATCH"
        if flag:
            bad.append((row, col, "%.4f vs %.3f" % (s, p)))
        print("%-15s %-22s %-11s %-9.3f %-9.4f %+7.4f %s%s" % (
            row, cell, col, p, s, d, samp, flag))
print("-" * 92)
print("mismatches/missing:", len(bad))
for b in bad:
    print("  ", b)
