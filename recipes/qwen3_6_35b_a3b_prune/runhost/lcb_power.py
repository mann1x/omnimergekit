"""Power analysis for the LCB-77q routing contrast BEFORE spending GPU time.

Greedy is non-viable for this family (eval/models/qwen3_6.yaml), so variance can only be
bought down with draws. Question: how many draws per arm (k) are needed before a routing
effect of a given size is separable from the churn we measured?

Model: each problem i has a latent pass-rate p_i under the `recommended` sampler. Estimate
p_i from the observed draws. Then simulate the PAIRED per-problem statistic
    d_i = p_hat_i(armB) - p_hat_i(armA),  stat = mean_i(d_i)
under H0 (both arms share p_i, i.e. routing does nothing) to get the null band, which is
the minimum detectable effect (MDE) at that k.

Note on the estimator: with only 2-3 observations per problem, p_hat is coarse. Laplace
smoothing keeps 0/3 and 3/3 problems from being treated as perfectly deterministic --
without it the null band is optimistically narrow.
"""
import json, os, random

random.seed(20260821)

CELLS_T8 = [
    "/srv/ml/eval_results/qwen_suite/lcb_v6_77q/qwenhybridp24_q6k",
    "/srv/ml/eval_results_routing/qwen_suite/lcb_v6_77q/armJ_t8_a",
    "/srv/ml/eval_results_routing/qwen_suite/lcb_v6_77q/armJ_t8_b",
]
CELLS_T10 = [
    "/srv/ml/eval_results_routing/qwen_suite/lcb_v6_77q/armJ_t10_a",
    "/srv/ml/eval_results_routing/qwen_suite/lcb_v6_77q/armJ_t10_b",
]

def load(d):
    s = json.load(open(os.path.join(d, "summary.json")))
    f = s["samples_file"]
    return {json.loads(l)["task_id"]: bool(json.loads(l)["passed"])
            for l in open(f) if l.strip()}

def gather(dirs):
    obs = {}
    used = 0
    for d in dirs:
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue
        used += 1
        for tid, ok in load(d).items():
            obs.setdefault(tid, []).append(ok)
    return obs, used

o8, n8 = gather(CELLS_T8)
o10, n10 = gather(CELLS_T10)
print("draws available: t8=%d  t10=%d" % (n8, n10))
ids = sorted(set(o8) & set(o10))
print("problems: %d" % len(ids))

# Two estimators of the latent rate, pooled over BOTH arms (H0: routing does nothing).
# At n=3 obs/problem these BRACKET the truth and neither alone is trustworthy:
#   laplace -- (s+1)/(n+2). Conservative: a 3/3 problem becomes p=0.8, so it still
#              contributes variance. Correct in spirit (3 draws cannot prove p=1) but it
#              pins every problem into [0.2,0.8] and thereby ERASES the stable stratum.
#   mle     -- s/n. Optimistic: a 3/3 problem becomes p=1.0, contributing ZERO variance,
#              which assumes away exactly the uncertainty we are trying to measure.
# The real MDE is between the two columns.
EST = {}
for i in ids:
    v = o8[i] + o10[i]
    s, n = sum(v), len(v)
    EST[i] = {"laplace": (s + 1.0) / (n + 2.0), "mle": s / n}

for est in ("laplace", "mle"):
    det = sum(1 for i in ids if EST[i][est] < 0.15 or EST[i][est] > 0.85)
    print("%-8s near-deterministic (<.15 or >.85): %d / %d" % (est, det, len(ids)))
print()
p = {}  # set per-estimator below

def simulate(k, est, trials=4000, subset=None):
    pool = subset or ids
    stats = []
    for _ in range(trials):
        tot = 0.0
        for i in pool:
            pi = EST[i][est]
            a = sum(random.random() < pi for _ in range(k)) / k
            b = sum(random.random() < pi for _ in range(k)) / k
            tot += (b - a)
        stats.append(tot / len(pool))
    stats.sort()
    return stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))]

def mde(k, est, subset=None, scale=1.0):
    lo, hi = simulate(k, est, subset=subset)
    return (hi - lo) / 2 * 100 * scale

print("MDE = half-width of the H0 95% band on the mean paired per-problem rate difference.")
print("An observed |effect| must EXCEED this to be separable from churn.")
print()
print("%-4s  %-12s %-12s  %s" % ("k", "conservative", "optimistic", "cost/arm (h)"))
print("%-4s  %-12s %-12s  %s" % ("", "(laplace)", "(mle)", ""))
for k in (1, 2, 4, 6, 8):
    print("%-4d  +/-%-9.2f +/-%-9.2f  %.1f" % (k, mde(k, "laplace"), mde(k, "mle"), k * 2.7))
print()

# Stratification is only worth costing under the estimator that can actually SEE a stable
# stratum. Under laplace at n=3 the stratum is all 77 problems, so it saves nothing --
# that is the honest readout, not a bug to tune away.
for est in ("laplace", "mle"):
    uns = [i for i in ids if 0.15 <= EST[i][est] <= 0.85]
    share = len(uns) / len(ids)
    print("stratified [%s]: extra draws on the %d/%d non-deterministic problems" % (
        est, len(uns), len(ids)))
    if len(uns) == len(ids):
        print("   -> stratum == whole bench; NO saving available under this estimator")
    else:
        for k in (4, 6, 8, 12):
            print("   k=%-3d full-bench-equiv MDE +/-%.2f pp   cost/arm %.1f h" % (
                k, mde(k, est, subset=uns, scale=share), k * 2.7 * share))
    print()
print("observed effect to beat: t8 -> t10 was -5.19 pp (draw A, single draw each)")
