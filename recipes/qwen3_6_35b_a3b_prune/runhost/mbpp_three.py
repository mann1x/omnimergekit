"""MBPP-500 three-way paired analysis: t8, t8r (same-config repeat), t10.

The t8-vs-t8r pair IS the engine-jitter band on armJ. A small SCORE difference does not
prove a small band -- net can hide churn (the LCB lesson). So compare CHURN, not just net.
"""
import json, os
from math import comb
R = "/srv/ml/eval_results_routing/qwen_suite/mbpp_full"

def load(c):
    s = json.load(open(os.path.join(R, c, "summary.json")))
    d = {}
    for ln in open(s["samples_file"]):
        ln = ln.strip()
        if ln:
            j = json.loads(ln)
            d[j["doc_id"]] = (bool(j["pass_at_1"]), j.get("doc_hash"))
    return d, s["score"]

def p_mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))

cells = {}
for c in ("armJ_g_t8", "armJ_g_t8r", "armJ_g_t10"):
    cells[c] = load(c)
    print("%-12s score=%.4f  n=%d  passes=%d" % (
        c, cells[c][1], len(cells[c][0]), sum(v[0] for v in cells[c][0].values())))
print()

def pair(x, y, label):
    dx, sx = cells[x]; dy, sy = cells[y]
    ids = sorted(set(dx) & set(dy))
    bad = [i for i in ids if dx[i][1] != dy[i][1]]
    pf = sum(1 for i in ids if dx[i][0] and not dy[i][0])
    fp = sum(1 for i in ids if not dx[i][0] and dy[i][0])
    p = p_mcnemar(pf, fp)
    print("%-34s  n=%d hashmismatch=%d" % (label, len(ids), len(bad)))
    print("    %.4f -> %.4f  = %+.2f pp | net %+d | CHURN %d (%d lost, %d gained) | net/churn %s"
          % (sx, sy, (sy - sx) * 100, fp - pf, pf + fp, pf, fp,
             "n/a" if pf + fp == 0 else "%.2f" % (abs(fp - pf) / (pf + fp))))
    print("    McNemar exact two-sided p = %.6f   %s" % (
        p, "NOT distinguishable from noise" if p > 0.05 else "*** SIGNIFICANT ***"))
    print()
    return pf + fp

band_churn = pair("armJ_g_t8", "armJ_g_t8r", "BAND  t8 -> t8r (same config)")
e1 = pair("armJ_g_t8",  "armJ_g_t10", "EFFECT t8  -> t10 (routing)")
e2 = pair("armJ_g_t8r", "armJ_g_t10", "EFFECT t8r -> t10 (routing)")
print("churn ratio: routing churn / same-config churn = %.1fx and %.1fx" % (
    e1 / band_churn if band_churn else float("inf"),
    e2 / band_churn if band_churn else float("inf")))
