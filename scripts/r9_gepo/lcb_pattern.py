"""Is gepo2's LCB deficit structured, or one bad sampled draw?

Checks: (1) failure OVERLAP across arms -- is there a shared hard core, or does each arm
fail a different scatter (the signature of draw noise)? (2) position/difficulty trend.
(3) whether the same arm pair ALSO diverges on the greedy cohort's problem set -- if the
problems gepo2 loses under sampling are ones it WON under greedy, the difference is the
draw, not the weights.
"""
import json, os

QS = "/srv/ml/eval_results/qwen_suite/lcb_v6_77q"
GR = "/srv/ml/eval_results/ream_arms/lcb_v6_77q_48k"
S = [("armJ", "qwenhybridp24_q6k"), ("gepo1", "qwena3bgepo1_q6k"), ("gepo2", "qwena3bgepo2_q6k")]
G = [("armJ", "lcb48k_armJ_hybrid_p24"), ("gepo1", "lcb48k_a3b_gepo1"), ("gepo2", "lcb48k_a3b_gepo2")]


def load(root, n):
    out = {}
    with open(os.path.join(root, n, "lcb_result.samples.jsonl")) as fh:
        for l in fh:
            if l.strip():
                d = json.loads(l)
                out[d.get("task_id") or d.get("doc_id")] = (bool(d.get("passed")), d.get("finish_reason"))
    return out


s = {t: load(QS, n) for t, n in S}
g = {t: load(GR, n) for t, n in G}
ids = sorted(s["armJ"])

print("=== failure OVERLAP, sampled cohort (n=77) ===")
F = {t: {k for k in ids if not s[t][k][0]} for t, _ in S}
for t, _ in S:
    print("  %-6s fails %d" % (t, len(F[t])))
print("  all three fail      : %d  (shared hard core)" % len(F["armJ"] & F["gepo1"] & F["gepo2"]))
print("  gepo1 & gepo2 only  : %d" % len((F["gepo1"] & F["gepo2"]) - F["armJ"]))
print("  gepo2 ONLY          : %d" % len(F["gepo2"] - F["armJ"] - F["gepo1"]))
print("  gepo1 ONLY          : %d" % len(F["gepo1"] - F["armJ"] - F["gepo2"]))
print("  armJ  ONLY          : %d" % len(F["armJ"] - F["gepo1"] - F["gepo2"]))

print("\n=== position trend (LCB ids are ~chronological = proxy for recency/difficulty) ===")
half = len(ids) // 2
for t, _ in S:
    a = sum(1 for k in ids[:half] if s[t][k][0]); b = sum(1 for k in ids[half:] if s[t][k][0])
    print("  %-6s first-half %2d/%2d   second-half %2d/%2d" % (t, a, half, b, len(ids) - half))

print("\n=== CROSS-COHORT: the 11 problems gepo2 loses to armJ under SAMPLING —")
print("    how did those same problems go under GREEDY? ===")
disc = sorted(k for k in ids if s["armJ"][k][0] and not s["gepo2"][k][0])
both = [k for k in disc if k in g["armJ"] and k in g["gepo2"]]
gw = sum(1 for k in both if g["gepo2"][k][0] and not g["armJ"][k][0])
gl = sum(1 for k in both if g["armJ"][k][0] and not g["gepo2"][k][0])
ge = sum(1 for k in both if g["armJ"][k][0] == g["gepo2"][k][0])
print("  of %d: greedy AGREED (gepo2 also lost) = %d | greedy REVERSED (gepo2 won) = %d | tied = %d"
      % (len(both), gl, gw, ge))

print("\n=== per-arm agreement between the two cohorts (same weights, diff sampler) ===")
for t, gn in [("armJ", "armJ"), ("gepo1", "gepo1"), ("gepo2", "gepo2")]:
    common = [k for k in ids if k in g[t]]
    agree = sum(1 for k in common if s[t][k][0] == g[t][k][0])
    flip_sg = sum(1 for k in common if s[t][k][0] and not g[t][k][0])
    flip_gs = sum(1 for k in common if g[t][k][0] and not s[t][k][0])
    print("  %-6s n=%d  same verdict=%d (%.0f%%)  sampled-only-pass=%d  greedy-only-pass=%d"
          % (t, len(common), agree, agree / len(common) * 100, flip_sg, flip_gs))
