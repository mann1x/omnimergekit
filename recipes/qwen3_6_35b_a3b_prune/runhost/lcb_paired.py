"""Paired per-problem LCB contingency.

A net score delta hides CHURN. 5 problems net can be 5 one-way drops (systematic)
or 12-and-7 (noise). This builds the 2x2 per pair and reports both, plus the
exact-binomial p on the discordant cells (McNemar), which is the only honest
read on whether a paired flip count is distinguishable from coin-flips.
"""
import json, os, sys
from math import comb

CELLS = {
    "banked_t8": "/srv/ml/eval_results/qwen_suite/lcb_v6_77q/qwenhybridp24_q6k",
    "fresh_t8":  "/srv/ml/eval_results_routing/qwen_suite/lcb_v6_77q/armJ_t8_a",
    "fresh_t10": "/srv/ml/eval_results_routing/qwen_suite/lcb_v6_77q/armJ_t10_a",
}

def load(cell_dir):
    """Use the samples file summary.json ACTUALLY points at -- the banked cell has both a
    pre- and post-b606 rescore file on disk and picking the wrong one is a silent basis mix."""
    s = json.load(open(os.path.join(cell_dir, "summary.json")))
    f = s.get("samples_file") or os.path.join(cell_dir, "lcb_result.samples.jsonl")
    if not os.path.exists(f):
        f = os.path.join(cell_dir, os.path.basename(f))
    out, toks = {}, {}
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d["task_id"]] = bool(d["passed"])
        toks[d["task_id"]] = d.get("completion_tokens")
    return out, toks, os.path.basename(f), s["score"]

def mcnemar_p(b, c):
    """Two-sided exact binomial on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)

data = {}
for name, d in CELLS.items():
    try:
        data[name] = load(d)
    except Exception as e:
        print("SKIP %s: %s" % (name, e)); continue

for name, (p, _t, fn, sc) in data.items():
    print("%-10s n=%-4d passes=%-4d score=%.4f  file=%s" % (
        name, len(p), sum(p.values()), sc, fn))
print()

def pair(a, b):
    if a not in data or b not in data:
        return
    pa, ta, _, sca = data[a]
    pb, tb, _, scb = data[b]
    ids = sorted(set(pa) & set(pb))
    only = (set(pa) ^ set(pb))
    pp = sum(1 for i in ids if pa[i] and pb[i])
    pf = sum(1 for i in ids if pa[i] and not pb[i])
    fp = sum(1 for i in ids if not pa[i] and pb[i])
    ff = sum(1 for i in ids if not pa[i] and not pb[i])
    print("=== %s -> %s   (paired on %d problems%s)" % (
        a, b, len(ids), "" if not only else "; %d id-mismatch EXCLUDED" % len(only)))
    print("    pass->pass %3d   pass->FAIL %3d" % (pp, pf))
    print("    FAIL->pass %3d   fail->fail %3d" % (fp, ff))
    print("    net = %+d problems (%.4f -> %.4f = %+.2f pp)" % (
        fp - pf, sca, scb, (scb - sca) * 100))
    print("    CHURN = %d flips total (%d down, %d up); net/churn = %s" % (
        pf + fp, pf, fp, "n/a" if pf + fp == 0 else "%.2f" % (abs(fp - pf) / (pf + fp))))
    print("    McNemar exact two-sided p = %.4f  %s" % (
        mcnemar_p(pf, fp),
        "<- NOT distinguishable from noise" if mcnemar_p(pf, fp) > 0.05 else "<- significant"))
    flips = [i for i in ids if pa[i] != pb[i]]
    print("    flipped: %s" % ", ".join(i.split("/")[-1] for i in sorted(flips)))
    print()

pair("banked_t8", "fresh_t8")   # SAME nominal config -> pure repeat noise
pair("fresh_t8", "fresh_t10")   # the routing contrast
