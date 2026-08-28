"""Corrected length analysis: reasoning channel + content, from omk's reasoning_log sidecar.

WHY: samples.jsonl `resps` holds CONTENT ONLY (Fix-A: content-only, reasoning only when
content is empty). Measuring token length off `resps` on a _think bench measures the ANSWER,
not the thinking -- which is precisely the channel a brevity intervention targets. The
sidecar records content_chars and reasoning_chars per doc; total = both.
"""
import json, os
R = "/srv/ml/eval_results/qwen_suite"
BENCH = ["gpqa_diamond_full", "gsm8k_100_boxed", "arc_challenge_100", "humaneval_full_think"]
ARMS = [("armJ", "qwenhybridp24_q6k"), ("gepo1", "qwena3bgepo1_q6k"), ("gepo2", "qwena3bgepo2_q6k")]


def med(x):
    x = sorted(x)
    return x[len(x) // 2] if x else 0


for b in BENCH:
    print("=== %s ===" % b)
    print("%-7s %5s %11s %11s %11s %11s %11s %8s"
          % ("arm", "n", "reas_p50", "reas_mean", "cont_p50", "cont_mean", "TOTAL_mean", "n_reas>0"))
    base = None
    for tag, name in ARMS:
        p = "%s/%s/%s/reasoning_log.jsonl" % (R, b, name)
        if not os.path.exists(p):
            print("%-7s ABSENT" % tag)
            continue
        rs, cs = [], []
        for line in open(p):
            if not line.strip():
                continue
            d = json.loads(line)
            rs.append(d.get("reasoning_chars") or 0)
            cs.append(d.get("content_chars") or 0)
        tot = sum(rs) / len(rs) + sum(cs) / len(cs)
        if base is None:
            base = tot
        print("%-7s %5d %11d %11.0f %11d %11.0f %11.0f %8d"
              % (tag, len(rs), med(rs), sum(rs) / len(rs), med(cs), sum(cs) / len(cs),
                 tot, sum(1 for v in rs if v > 0)))
    print()
