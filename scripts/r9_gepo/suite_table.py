"""Out-of-band corrected suite table for the gepo2 qwen_suite cohort.

WHY THIS EXISTS: suite_gepo2.sh's end-of-run table reads summary.json .score for both
arms with no basis check, so it prints a CROSS-BASIS gsm8k row (bug-633) -- armJ's
summary selected filter=boxed while gepo1/gepo2 selected flexible-extract. That row
shows a phantom +1pp gepo2 win that is purely a filter artifact. The running script is
NOT edited; this reproduces its table with the basis enforced.

Per bench: every arm must agree on (metric, filter) AND on sampler. Where they disagree,
fall back to the raw results file and recompute on every basis COMMON to all arms, and
label the row CORRECTED. Where no common basis exists, the row is VOID -- never a number.
"""
import glob, json, os, sys

RES = "/srv/ml/eval_results/qwen_suite"
ARMS = ["qwenhybridp24_q6k", "qwena3bgepo1_q6k", "qwena3bgepo2_q6k"]
LBL = {"qwenhybridp24_q6k": "armJ", "qwena3bgepo1_q6k": "gepo1", "qwena3bgepo2_q6k": "gepo2"}
# bug-630: multipl_e's canonical armJ comparator is the _b604 RE-RUN; the un-suffixed
# cell is the VOID pre-scorer-fix one and is exactly what a cloned driver picks up.
OVERRIDE = {"multipl_e_100": {"qwenhybridp24_q6k": "qwenhybridp24_q6k_b604"}}
BENCHES = ["gpqa_diamond_full", "gsm8k_100_boxed", "arc_challenge_100",
           "humaneval_full_think", "humanevalplus_full_think", "multipl_e_100",
           "ifeval_100", "math500_100_qwen", "aime_30_qwen", "lcb_v6_77q"]
WANT_SAMPLER = "recommended"


def cell(bench, arm):
    return os.path.join(RES, bench, OVERRIDE.get(bench, {}).get(arm, arm))


def raw_metrics(d):
    """-> {'metric,filter': score} from the lm-eval results file, {} if not lm-eval."""
    f = glob.glob(d + "/**/results_*.json", recursive=True)
    if not f:
        return {}
    j = json.load(open(sorted(f)[-1]))
    out = {}
    for _task, m in (j.get("results") or {}).items():
        for k, v in m.items():
            if isinstance(v, float) and not k.startswith(("alias", "exact_match_stderr")) \
                    and "_stderr" not in k and "," in k:
                out[k] = v
    return out


rows, notes = [], []
for b in BENCHES:
    got = {}
    for a in ARMS:
        p = cell(b, a) + "/summary.json"
        if not os.path.exists(p):
            got[a] = None
            continue
        s = json.load(open(p))
        got[a] = (s.get("score"), s.get("metric"), s.get("filter"),
                  ((s.get("sampler") or {}).get("name")), cell(b, a))

    if any(v is None for v in got.values()):
        rows.append((b, "PENDING", {a: (None if got[a] is None else got[a][0]) for a in ARMS}, ""))
        continue

    samp = {got[a][3] for a in ARMS}
    if samp != {WANT_SAMPLER}:
        rows.append((b, "VOID", {a: None for a in ARMS}, "sampler mismatch %s" % samp))
        notes.append("%s VOID: sampler=%s (want %s) -- cohorts must never be pooled" % (b, samp, WANT_SAMPLER))
        continue

    basis = {got[a][1:3] for a in ARMS}
    if len(basis) == 1:
        m, f = list(basis)[0]
        rows.append((b, "OK", {a: got[a][0] for a in ARMS}, "%s,%s" % (m, f)))
        continue

    # disagreement -> recompute on every basis common to all arms
    common = None
    per = {}
    for a in ARMS:
        per[a] = raw_metrics(got[a][4])
        common = set(per[a]) if common is None else (common & set(per[a]))
    common = sorted(common or [])
    notes.append("%s CROSS-BASIS (bug-633): arms disagree %s -- summary .score is NOT comparable"
                 % (b, sorted("%s,%s" % x for x in basis)))
    for a in ARMS:
        notes.append("      %-6s summary picked %s,%s = %.4f" % (LBL[a], got[a][1], got[a][2], got[a][0]))
    if not common:
        rows.append((b, "VOID", {a: None for a in ARMS}, "no common basis"))
        continue
    for k in common:
        rows.append((b, "CORRECTED", {a: per[a][k] for a in ARMS}, k))

print("%-26s %-10s %8s %8s %8s   %s" % ("bench", "basis", "armJ", "gepo1", "gepo2", "metric,filter"))
print("-" * 92)
for b, st, sc, meta in rows:
    def fmt(a):
        v = sc.get(a)
        return "%8.4f" % v if isinstance(v, float) else "%8s" % ("--" if v is None else v)
    print("%-26s %-10s %s %s %s   %s" % (b, st, fmt(ARMS[0]), fmt(ARMS[1]), fmt(ARMS[2]), meta))

if notes:
    print("\n=== BASIS NOTES ===")
    for n in notes:
        print("  " + n)
