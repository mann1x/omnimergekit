"""Paired per-problem MBPP-500 contingency, t8 vs t10 (greedy cohort).

MBPP t10 read 5.2pp below t8 (0.732 vs 0.784) = 26 problems of 500. Truncation is already
ruled out as the mechanism (1/500 vs 3/500 capped). The question this answers: is the drop
SYSTEMATIC (mostly one-way losses) or CHURN (large two-way flips netting to 26)?

Verifies doc_hash equality first -- a paired test on non-identical problems is meaningless.
"""
import json, os
from math import comb

R = "/srv/ml/eval_results_routing/qwen_suite/mbpp_full"

def load(cell):
    s = json.load(open(os.path.join(R, cell, "summary.json")))
    out = {}
    for line in open(s["samples_file"]):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d["doc_id"]] = (bool(d["pass_at_1"]), d.get("doc_hash"),
                            (d.get("filtered_resps") or [[""]])[0][0] if d.get("filtered_resps") else "")
    return out, s["score"]

def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))

a, sa = load("armJ_g_t8")
b, sb = load("armJ_g_t10")
ids = sorted(set(a) & set(b))
print("paired on %d docs (t8 n=%d, t10 n=%d)" % (len(ids), len(a), len(b)))

# BASIS CHECK: identical problems, or the whole comparison is void.
mismatch = [i for i in ids if a[i][1] != b[i][1]]
print("doc_hash mismatches: %d  %s" % (len(mismatch), "OK" if not mismatch else "*** BASIS BREAK ***"))

pp = sum(1 for i in ids if a[i][0] and b[i][0])
pf = sum(1 for i in ids if a[i][0] and not b[i][0])
fp = sum(1 for i in ids if not a[i][0] and b[i][0])
ff = sum(1 for i in ids if not a[i][0] and not b[i][0])
print()
print("            t10 pass   t10 FAIL")
print("t8 pass   %8d %10d" % (pp, pf))
print("t8 FAIL   %8d %10d" % (fp, ff))
print()
print("score      t8=%.4f  t10=%.4f   delta=%+.2f pp" % (sa, sb, (sb - sa) * 100))
print("net        %+d problems" % (fp - pf))
print("CHURN      %d flips (%d lost, %d gained); net/churn = %s" % (
    pf + fp, pf, fp, "n/a" if pf + fp == 0 else "%.2f" % (abs(fp - pf) / (pf + fp))))
print("McNemar exact two-sided p = %.6f  %s" % (
    mcnemar_p(pf, fp),
    "<- NOT distinguishable from noise" if mcnemar_p(pf, fp) > 0.05 else "<- SIGNIFICANT"))
print()
# An empty/degenerate completion on the LOSING side points at a generation failure rather
# than a wrong-but-real answer; worth separating before calling this a capability drop.
lost = [i for i in ids if a[i][0] and not b[i][0]]
deg = [i for i in lost if len(b[i][2].strip()) < 20]
print("of the %d lost problems, %d have a <20-char t10 completion (degenerate/empty)" % (len(lost), len(deg)))
print("lost doc_ids: %s" % (", ".join(str(i) for i in lost[:40]) + (" ..." if len(lost) > 40 else "")))
