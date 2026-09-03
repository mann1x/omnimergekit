#!/usr/bin/env python
"""qwen_suite arm table: armJ vs gepo1/2/3 by default, plus any arms given on argv.

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
# EXTRA ARMS FROM ARGV: `table_gepo_arms.py gepo4=qwena3bgepo4_q6k [...]`. run4 is a
# 424-step epoch checkpointed every 2 steps, so the arm under test may well be
# `gepo4ck218` rather than a single final `gepo4` -- and adding each one by editing this
# file is exactly the "editing embedded Python" hazard the file was extracted to avoid.
# Defaults are unchanged when no argument is given.
for _a in sys.argv[1:]:
    if "=" not in _a:
        sys.exit(f"REFUSE: extra arm {_a!r} must be label=served_name")
    _lab, _nm = _a.split("=", 1)
    if _lab in [x for x, _ in ARMS]:
        sys.exit(f"REFUSE: arm label {_lab!r} already present; pick a distinct label so "
                 "two different models cannot collide in one column")
    ARMS.append((_lab, _nm))
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
# ARM COUNT IS DYNAMIC. The header/row formats used to hardcode 4 arms and 3 delta
# columns, so passing TWO extra arms (the run4 dose series needs ck60 AND ck218 in one
# table) died with "not all arguments converted during string formatting". Widths are 10,
# not 8, because a label like "gepo4ck218" is 10 chars and would otherwise push the
# header out of step with the %8.4f data rows -- a silently misread table.
nA = len(labels)
# Token columns compare armJ against the NEWEST arm; with no argv that is gepo3, which is
# exactly what this script did before.
TOKREF = labels[-1]
hdr = ("%-26s" + " %10s" * nA + "   " + " %10s" * (nA - 1) + "   %10s %12s %8s") % (
    "bench", *labels, *labels[1:], "armJ tok", "%s tok" % TOKREF, "tok %")
print(hdr)
print("-" * len(hdr))
print("(the %d columns after the gap are DELTAS vs armJ, in percentage points)" % (nA - 1))

acc = {a: 0.0 for a in labels}
tot = {"armJ": 0, TOKREF: 0}
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
    ta, t3 = toks(ds["armJ"]), toks(ds[TOKREF])
    tp = ("%+.1f%%" % ((t3 - ta) / ta * 100)) if ta else "-"
    base = ds["armJ"]["score"]
    print(("%-26s" + " %10.4f" * nA + "   " + " %+10.2f" * (nA - 1) + "   %10d %12d %8s") % (
        b, *[ds[a]["score"] for a in labels],
        *[(ds[a]["score"] - base) * 100 for a in labels[1:]], ta, t3, tp))
    for a in labels:
        acc[a] += ds[a]["score"]
    tot["armJ"] += ta
    tot[TOKREF] += t3
    n += 1

if n:
    print("-" * len(hdr))
    base = acc["armJ"] / n
    print(("%-26s" + " %10.4f" * nA + "   " + " %+10.2f" * (nA - 1) + "   %10d %12d %8s") % (
        "MEAN over %d" % n, *[acc[a] / n for a in labels],
        *[(acc[a] - acc["armJ"]) / n * 100 for a in labels[1:]],
        tot["armJ"], tot[TOKREF],
        ("%+.1f%%" % ((tot[TOKREF] - tot["armJ"]) / tot["armJ"] * 100)) if tot["armJ"] else "-"))

print()
if bad_sampler:
    print("!!! SAMPLER MISMATCH -- these cells are NOT in this cohort and the row above is void:")
    for x in bad_sampler:
        print("      " + x)
elif n == 0:
    # A vacuous pass. The sampler check inspects only COMPLETE rows, so with none read it
    # has verified nothing -- printing the reassuring line here would certify absence.
    print("sampler: NOT CHECKED -- 0 complete rows; every bench above is INCOMPLETE.")
else:
    print("sampler: all %d complete rows recorded '%s' -- one cohort, differences are "
          "tableable." % (n, WANT_SAMPLER))

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
