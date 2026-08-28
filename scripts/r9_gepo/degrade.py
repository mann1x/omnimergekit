"""GEPO degradation census: armJ -> gepo1 -> gepo2, paired per problem.

WHY: the suite table gives per-bench scores, but a score delta at n=77..198 can
be a handful of problems. To decide WHICH capabilities the brevity training
actually eroded -- and therefore what a replay set must protect -- we need, per
bench: the paired discordance (armJ-only vs gepo2-only wins), McNemar's exact
p, and whether the movement is MONOTONE across gepo1->gepo2 (a dose-response,
which is far stronger evidence than one arm's delta).

Pairing key is doc_id (lm-eval) / task_id (LCB). Filter+metric per bench are
pinned to the SAME (metric, filter) the canonical summary selects, so these
numbers reconcile with suite_table.py rather than forming a second basis.

Length is reported in CHARS here only as a secondary signal and is explicitly
labelled as such -- the token-accurate channel is LCB's completion field
(bug-638: chars and tokens can disagree in DIRECTION).
"""
import json
import os
import sys
from math import comb

RES = "/srv/ml/eval_results/qwen_suite"
ARMS = [("armJ", "qwenhybridp24_q6k"),
        ("gepo1", "qwena3bgepo1_q6k"),
        ("gepo2", "qwena3bgepo2_q6k")]

# bench -> (metric_key, filter_value)  matching the canonical summary selection
LM_BENCHES = {
    "gpqa_diamond_full":        ("exact_match", "flexible-extract"),
    "humaneval_full_think":     ("pass@1", "extract_chat"),
    "humanevalplus_full_think": ("pass@1", "extract_chat"),
    "ifeval_100":               ("prompt_level_strict_acc", "none"),
    "arc_challenge_100":        ("exact_match", "none"),
    "math500_100_qwen":         ("math_verify", "none"),
    "aime_30_qwen":             ("exact_match", "none"),
    "gsm8k_100_boxed":          ("exact_match", "flexible-extract"),
}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def load_lm(bench, cell, metric, filt):
    """-> {doc_id: (correct: bool, resp_chars: int)} or None"""
    d = os.path.join(RES, bench, cell, "lm_eval_out")
    if not os.path.isdir(d):
        return None
    files = []
    for root, _, fns in os.walk(d):
        for fn in fns:
            if fn.startswith("samples_") and fn.endswith(".jsonl"):
                files.append(os.path.join(root, fn))
    if not files:
        return None
    files.sort()
    out = {}
    for f in files:
        for line in open(f, errors="ignore"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("filter") != filt:
                continue
            if metric not in r:
                continue
            v = r[metric]
            ok = bool(v) if isinstance(v, bool) else (float(v) >= 0.5)
            resps = r.get("resps") or [[""]]
            txt = resps[0][0] if resps and resps[0] else ""
            out[r["doc_id"]] = (ok, len(txt or ""))
    return out or None


def load_lcb(cell):
    p = os.path.join(RES, "lcb_v6_77q", cell, "lcb_result.samples.jsonl")
    if not os.path.isfile(p):
        return None
    out = {}
    for line in open(p, errors="ignore"):
        if line.strip():
            r = json.loads(line)
            out[r.get("task_id") or r.get("doc_id")] = (
                bool(r.get("passed")), len(r.get("completion") or ""))
    return out or None


rows = []
for bench, (metric, filt) in LM_BENCHES.items():
    arms = {t: load_lm(bench, c, metric, filt) for t, c in ARMS}
    if any(v is None for v in arms.values()):
        missing = [t for t, v in arms.items() if v is None]
        print(f"  SKIP {bench}: missing {missing}")
        continue
    rows.append((bench, arms))
lcb = {t: load_lcb(c) for t, c in ARMS}
if all(v is not None for v in lcb.values()):
    rows.append(("lcb_v6_77q", lcb))

print(f"{'bench':<26} {'n':>4} {'armJ':>6} {'gepo1':>6} {'gepo2':>6} "
      f"{'d(g2-aJ)':>9} {'aJonly':>7} {'g2only':>7} {'p':>7}  {'monotone?':<10}")
print("-" * 108)
summary = []
for bench, arms in rows:
    common = sorted(set(arms["armJ"]) & set(arms["gepo1"]) & set(arms["gepo2"]),
                    key=str)
    n = len(common)
    if not n:
        continue
    pa = sum(arms["armJ"][k][0] for k in common)
    p1 = sum(arms["gepo1"][k][0] for k in common)
    p2 = sum(arms["gepo2"][k][0] for k in common)
    b = sum(1 for k in common if arms["armJ"][k][0] and not arms["gepo2"][k][0])
    c = sum(1 for k in common if arms["gepo2"][k][0] and not arms["armJ"][k][0])
    p = mcnemar(b, c)
    d = (p2 - pa) / n * 100
    mono = "MONOTONE-DN" if pa > p1 > p2 else ("MONOTONE-UP" if pa < p1 < p2 else "-")
    print(f"{bench:<26} {n:>4} {pa/n:>6.4f} {p1/n:>6.4f} {p2/n:>6.4f} "
          f"{d:>+8.2f}pp {b:>7} {c:>7} {p:>7.4f}  {mono:<10}")
    summary.append((bench, n, d, b, c, p, mono, arms, common))

print()
print("=== DEGRADED benches (gepo2 below armJ), ranked by |problems lost| ===")
deg = [s for s in summary if s[2] < 0]
deg.sort(key=lambda s: s[3] - s[4], reverse=True)
for bench, n, d, b, c, p, mono, arms, common in deg:
    net = b - c
    print(f"  {bench:<26} net -{net:<3} problems  ({d:+.2f}pp, n={n})  "
          f"discordant {b}/{c}  p={p:.4f}  {mono}")

print()
print("=== length signal on DEGRADED benches (CHARS -- secondary, see bug-638) ===")
print(f"  {'bench':<26} {'armJ p50':>9} {'gepo1 p50':>10} {'gepo2 p50':>10} {'g2/aJ':>7}")
for bench, n, d, b, c, p, mono, arms, common in deg:
    def p50(t):
        v = sorted(arms[t][k][1] for k in common)
        return v[len(v) // 2]
    a, g1, g2 = p50("armJ"), p50("gepo1"), p50("gepo2")
    print(f"  {bench:<26} {a:>9} {g1:>10} {g2:>10} {g2/a if a else 0:>6.2f}x")

print()
print("=== the problems gepo2 LOST (armJ pass -> gepo2 fail), for replay mining ===")
for bench, n, d, b, c, p, mono, arms, common in deg:
    lost = [k for k in common
            if arms["armJ"][k][0] and not arms["gepo2"][k][0]]
    also1 = [k for k in lost if not arms["gepo1"][k][0]]
    print(f"  {bench:<26} lost {len(lost):>3}  "
          f"(of which gepo1 ALSO failed: {len(also1)} = persistent, "
          f"{len(lost)-len(also1)} = new in gepo2)")
    print(f"      ids: {[str(x) for x in lost[:20]]}")
