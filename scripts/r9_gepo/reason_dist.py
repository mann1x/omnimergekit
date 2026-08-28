"""Reasoning-channel length DISTRIBUTION across armJ/gepo1/gepo2.

WHY unpaired: the sidecar producer writes "idx": 0 on every record (bug-639) --
198 lines, all idx=0 -- so the reasoning channel cannot be joined to doc_id and
per-problem pairing is impossible. The distribution is still sound: identical
problem set, identical n, same order-of-completion bias in every arm. So this
answers "did the thinking get shorter overall?" but NOT "did it get shorter on
the problems it lost".

CHANNEL: lm-eval's `resps` is CONTENT ONLY on thinking benches -- the thinking
is stripped into this sidecar (bug-635). Measuring brevity off `resps` measures
the ANSWER and is the wrong channel entirely.

UNITS: chars, not tokens (the sidecar stores no token counts, bug-638). Ratios
within a bench are the claim; absolute values are not comparable across benches.
"""
import json
import os

RES = "/srv/ml/eval_results/qwen_suite"
ARMS = [("armJ", "qwenhybridp24_q6k"),
        ("gepo1", "qwena3bgepo1_q6k"),
        ("gepo2", "qwena3bgepo2_q6k")]
BENCHES = ["gpqa_diamond_full", "humaneval_full_think",
           "humanevalplus_full_think", "ifeval_100", "arc_challenge_100",
           "math500_100_qwen", "aime_30_qwen", "gsm8k_100_boxed"]


def load(bench, cell):
    p = os.path.join(RES, bench, cell, "reasoning_log.jsonl")
    if not os.path.isfile(p):
        return None
    vals = []
    for line in open(p, errors="ignore"):
        if line.strip():
            vals.append(int(json.loads(line).get("reasoning_chars", 0)))
    return vals or None


def pct(v, q):
    if not v:
        return 0
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * q))]


print("=== reasoning-channel chars, UNPAIRED distribution (the GEPO objective) ===")
print(f"{'bench':<26} {'n':>4} {'arm':<6} {'p50':>8} {'p90':>8} {'max':>8} "
      f"{'mean':>8} {'p50 vs armJ':>12}")
print("-" * 92)
for b in BENCHES:
    arms = {t: load(b, c) for t, c in ARMS}
    if any(v is None for v in arms.values()):
        continue
    if not any(sum(v) for v in arms.values()):
        print(f"{b:<26} (no thinking emitted on this bench)")
        continue
    base = pct(arms["armJ"], 0.5)
    for t, _ in ARMS:
        v = arms[t]
        r = (pct(v, 0.5) / base) if base else 0
        print(f"{b if t=='armJ' else '':<26} {len(v):>4} {t:<6} "
              f"{pct(v,0.5):>8} {pct(v,0.9):>8} {max(v):>8} "
              f"{sum(v)/len(v):>8.0f} {r:>11.2f}x")
    print()
