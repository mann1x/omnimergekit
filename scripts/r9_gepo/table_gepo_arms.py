#!/usr/bin/env python
"""Four-arm qwen_suite table: armJ vs gepo1 vs gepo2 vs gepo3, one cohort.

Extracted from suite_gepo2.sh's inline heredoc so a fourth arm could be added without
editing embedded Python inside a running shell script.

COHORT DISCIPLINE. Every cell here must record sampler=recommended. A greedy cell and a
sampled cell are different cohorts and differencing them is meaningless, so the sampler
is READ from each summary.json and a mismatch is printed as a loud row rather than
silently averaged. Scores come from summary.json .score -- never a raw results_*.json.
"""
import json
import os
import sys

R = "/srv/ml/eval_results/qwen_suite"
ARMS = [("armJ", "qwenhybridp24_q6k"), ("gepo1", "qwena3bgepo1_q6k"),
        ("gepo2", "qwena3bgepo2_q6k"), ("gepo3", "qwena3bgepo3_q6k")]
BENCH = ["gpqa_diamond_full", "gsm8k_100_boxed", "arc_challenge_100",
         "humaneval_full_think", "humanevalplus_full_think", "multipl_e_100",
         "ifeval_100", "math500_100_qwen", "aime_30_qwen", "lcb_v6_77q"]
# armJ's multipl_e comparator is the post-bug-604 RE-RUN cell, not the pre-fix one.
OVERRIDE = {"multipl_e_100": {"armJ": "qwenhybridp24_q6k_b604"}}
WANT_SAMPLER = "recommended"


def cell(bench, arm, name):
    return OVERRIDE.get(bench, {}).get(arm, name)


def load(bench, arm, name):
    try:
        return json.load(open(os.path.join(R, bench, cell(bench, arm, name), "summary.json")))
    except Exception:
        return None


def toks(d):
    return ((d.get("token_stats") or {}).get("completion_tokens") or {}).get("sum") or 0


def sampler(d):
    return ((d.get("sampler") or {}).get("name")) or "NONE"


labels = [a for a, _ in ARMS]
hdr = ("%-26s" + " %8s" * 4 + "   " + " %8s" * 3 + "   %10s %10s %8s") % (
    "bench", *labels, "g1-armJ", "g2-armJ", "g3-armJ", "armJ tok", "gepo3 tok", "tok %")
print(hdr)
print("-" * len(hdr))

acc = {a: 0.0 for a in labels}
tot = {"armJ": 0, "gepo3": 0}
n = 0
bad_sampler = []

for b in BENCH:
    ds = {a: load(b, a, nm) for a, nm in ARMS}
    if not all(ds.values()):
        have = " ".join(a for a in labels if ds[a])
        print("%-26s  INCOMPLETE (have: %s)" % (b, have or "none"))
        continue
    for a in labels:
        s = sampler(ds[a])
        if s != WANT_SAMPLER:
            bad_sampler.append("%s/%s sampler=%s" % (b, a, s))
    ta, t3 = toks(ds["armJ"]), toks(ds["gepo3"])
    tp = ("%+.1f%%" % ((t3 - ta) / ta * 100)) if ta else "-"
    base = ds["armJ"]["score"]
    print(("%-26s" + " %8.4f" * 4 + "   " + " %+8.2f" * 3 + "   %10d %10d %8s") % (
        b, *[ds[a]["score"] for a in labels],
        *[(ds[a]["score"] - base) * 100 for a in labels[1:]], ta, t3, tp))
    for a in labels:
        acc[a] += ds[a]["score"]
    tot["armJ"] += ta
    tot["gepo3"] += t3
    n += 1

if n:
    print("-" * len(hdr))
    base = acc["armJ"] / n
    print(("%-26s" + " %8.4f" * 4 + "   " + " %+8.2f" * 3 + "   %10d %10d %8s") % (
        "MEAN over %d" % n, *[acc[a] / n for a in labels],
        *[(acc[a] - acc["armJ"]) / n * 100 for a in labels[1:]],
        tot["armJ"], tot["gepo3"],
        ("%+.1f%%" % ((tot["gepo3"] - tot["armJ"]) / tot["armJ"] * 100)) if tot["armJ"] else "-"))

print()
if bad_sampler:
    print("!!! SAMPLER MISMATCH -- these cells are NOT in this cohort and the row above is void:")
    for x in bad_sampler:
        print("      " + x)
else:
    print("sampler: all cells recorded '%s' -- one cohort, differences are tableable." % WANT_SAMPLER)

print()
print("READING THIS TABLE")
print("  multipl_e_100 armJ column is the _b604 RE-RUN cell (post-bug-604 extractor).")
print("  SAMPLED cohort (temp 0.6): every cell is ONE draw.")
print("  The LCB draw-to-draw band is MEASURED, not assumed: a gepo2-vs-gepo2 repeat on")
print("  lcb_v6_77q gave 20 discordant problems, +5.19pp, McNemar p=0.5034 -- a paired SE")
print("  of sqrt(20)/77 = 5.8pp. An LCB difference below ~11-12pp is NOT resolvable at this")
print("  n. Treat the other single-draw benches as having their own unmeasured floors.")
print("  gepo3 also changed lora_dropout 0.05->0 (ParamWrapper constraint), so a POSITIVE")
print("  gepo3 needs the dropout-0 base-scope control before 'the router did it' is claimable.")
print("  Do NOT pool these rows with the greedy ream_arms/* cells.")
