"""Cross-cohort verdict flips, with the CAP removed as an explanation.

Restricting to problems untruncated in BOTH the greedy and the sampled run isolates the
sampler+draw from the cap. If flips stay ~20%, the per-problem verdict on LCB-77 is
simply not stable across a re-draw, and a 11-vs-5 discordant split cannot be read as a
weights difference without a same-basis repeat.
"""
import json, os
from math import comb

QS = "/srv/ml/eval_results/qwen_suite/lcb_v6_77q"
GR = "/srv/ml/eval_results/ream_arms/lcb_v6_77q_48k"
PAIR = [("armJ", "qwenhybridp24_q6k", "lcb48k_armJ_hybrid_p24"),
        ("gepo1", "qwena3bgepo1_q6k", "lcb48k_a3b_gepo1"),
        ("gepo2", "qwena3bgepo2_q6k", "lcb48k_a3b_gepo2")]


def load(root, n):
    out = {}
    with open(os.path.join(root, n, "lcb_result.samples.jsonl")) as fh:
        for l in fh:
            if l.strip():
                d = json.loads(l)
                out[d.get("task_id") or d.get("doc_id")] = (bool(d.get("passed")), d.get("finish_reason"))
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


print("%-7s %6s %8s %10s %10s  %8s" % ("arm", "n", "flips", "samp-only", "greedy-only", "p"))
print("-" * 56)
for tag, sn, gn in PAIR:
    S, G = load(QS, sn), load(GR, gn)
    keep = [k for k in S if k in G and S[k][1] != "length" and G[k][1] != "length"]
    a = sum(1 for k in keep if S[k][0] and not G[k][0])
    b = sum(1 for k in keep if G[k][0] and not S[k][0])
    print("%-7s %6d %8d %10d %10d  %8.4f"
          % (tag, len(keep), a + b, a, b, mcnemar(a, b)))
print("\n(both-untruncated only -- the generation cap cannot explain these flips)")
