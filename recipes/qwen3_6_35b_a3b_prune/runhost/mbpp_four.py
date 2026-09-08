"""Four-cell MBPP-500 paired analysis: two t8 draws vs two t10 draws (greedy cohort).

The point of the 4th cell: a release decision must not rest on a single draw of the LOSING
arm. With t10 drawn twice I can compute a t10 band the same way I computed the t8 band, and
check that the four cross-arm contrasts agree.

Verifies doc_hash equality across ALL cells first -- a paired test on non-identical problems
is meaningless. Also prints each cell's cap verdict + capped count, because a cap difference
between arms is a competing explanation for a score gap.
"""
import json, os, itertools
from math import comb

R = "/srv/ml/eval_results_routing/qwen_suite/mbpp_full"
CELLS = ["armJ_g_t8", "armJ_g_t8r", "armJ_g_t10", "armJ_g_t10r"]

def load(cell):
    s = json.load(open(os.path.join(R, cell, "summary.json")))
    per, capped = {}, 0
    for line in open(s["samples_file"]):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        fr = d.get("filtered_resps") or [[""]]
        txt = fr[0][0] if fr and fr[0] else ""
        per[d["doc_id"]] = (bool(d["pass_at_1"]), d.get("doc_hash"), txt)
    caps = s.get("generation_caps") or {}
    return per, s["score"], caps

def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))

D, S, C = {}, {}, {}
for c in CELLS:
    D[c], S[c], C[c] = load(c)

ids = sorted(set.intersection(*[set(D[c]) for c in CELLS]))
print("paired on %d docs common to all 4 cells" % len(ids))
bad = [i for i in ids if len({D[c][i][1] for c in CELLS}) != 1]
print("doc_hash mismatches across cells: %d  %s\n" % (len(bad), "OK" if not bad else "*** BASIS BREAK ***"))

print("%-14s %8s  %-9s  %s" % ("cell", "score", "verdict", "caps"))
for c in CELLS:
    print("%-14s %8.4f  %-9s  %s" % (c, S[c], C[c].get("verdict"),
          {k: v for k, v in C[c].items() if k != "verdict"}))
print()

def contrast(a, b, label):
    pf = sum(1 for i in ids if D[a][i][0] and not D[b][i][0])
    fp = sum(1 for i in ids if not D[a][i][0] and D[b][i][0])
    ch = pf + fp
    p = mcnemar_p(pf, fp)
    print("%-8s %-12s -> %-12s %+6.2f pp | net %+4d | churn %3d | net/churn %s | p = %.6f %s"
          % (label, a, b, (S[b] - S[a]) * 100, fp - pf, ch,
             "n/a " if ch == 0 else "%.2f" % (abs(fp - pf) / ch), p,
             "***" if p < 0.05 else ""))
    return ch

print("--- BANDS (same config, second draw) ---")
b8 = contrast("armJ_g_t8", "armJ_g_t8r", "BAND")
b10 = contrast("armJ_g_t10", "armJ_g_t10r", "BAND")
print("\n--- EFFECT (t8 arm vs t10 arm, all four cross pairs) ---")
eff = [contrast(a, b, "EFFECT") for a, b in itertools.product(["armJ_g_t8", "armJ_g_t8r"],
                                                              ["armJ_g_t10", "armJ_g_t10r"])]
bandmean = (b8 + b10) / 2.0
print("\nchurn: band mean %.1f  vs  effect mean %.1f  ->  %.1fx inflation under routing"
      % (bandmean, sum(eff) / len(eff), (sum(eff) / len(eff)) / bandmean if bandmean else 0))

# Arm-level: a problem is an ARM pass only if BOTH draws of that arm pass. This is the
# draw-robust view -- it strips problems that are simply unstable under batch numerics.
st8 = {i: D["armJ_g_t8"][i][0] and D["armJ_g_t8r"][i][0] for i in ids}
st10 = {i: D["armJ_g_t10"][i][0] and D["armJ_g_t10r"][i][0] for i in ids}
pf = sum(1 for i in ids if st8[i] and not st10[i])
fp = sum(1 for i in ids if not st8[i] and st10[i])
print("\n--- STABLE-PASS (passes BOTH draws of its arm) ---")
print("t8 stable %d/%d   t10 stable %d/%d   net %+d   churn %d   p = %.6f"
      % (sum(st8.values()), len(ids), sum(st10.values()), len(ids), fp - pf, pf + fp, mcnemar_p(pf, fp)))
