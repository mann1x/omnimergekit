"""Final paired readout: gepo2 LCB draw 2 vs the banked draw 1 — the within-arm noise floor.

Reads the CANONICAL score from summary.json (.score), never the log and never a raw
results_*.json — the log score is stale by construction and the raw keys are the
2026-05-23 false-alarm trap. The log is used ONLY to recover per-problem verdicts,
which summary.json does not carry, and the two are cross-checked against each other:
if the log-derived pass count disagrees with the canonical score, the pairing is not
describing the cell that was banked and nothing downstream is readable.

Also re-reads the SAMPLER of both cells. A greedy row and a sampled row can never be
pooled, and this comparison is only meaningful if both draws ran `recommended`.
"""
import json
import pathlib
import re
import sys
from math import comb

BANK1 = pathlib.Path("/srv/ml/eval_results/qwen_suite/lcb_v6_77q/qwena3bgepo2_q6k")
BANK2 = pathlib.Path("/srv/ml/eval_results/qwen_suite/lcb_v6_77q/qwena3bgepo2_q6k_rpt2")
LOG = pathlib.Path("/mnt/sdc/ml/brevity/gepo/lcb_repeat.log")


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def summary(d: pathlib.Path):
    p = d / "summary.json"
    if not p.is_file():
        sys.exit(f"REFUSE: no summary.json at {p} — the cell has not banked yet.")
    return json.loads(p.read_text())


s1, s2 = summary(BANK1), summary(BANK2)


def sampler_of(s):
    v = s.get("sampler")
    return (v or {}).get("name", "MISSING") if isinstance(v, dict) else str(v)


n1, n2 = sampler_of(s1), sampler_of(s2)
print("=== basis check (a cohort fact, read not assumed) ===")
print(f"  draw1 sampler={n1}  metric={s1.get('metric')}  filter={s1.get('filter')}")
print(f"  draw2 sampler={n2}  metric={s2.get('metric')}  filter={s2.get('filter')}")
if n1 != n2:
    sys.exit(f"REFUSE: sampler mismatch ({n1} vs {n2}) — these two draws are different "
             "cohorts and must never be pooled or differenced.")

print("\n=== canonical scores (summary.json .score) ===")
print(f"  draw1 {s1['score']:.4f}    draw2 {s2['score']:.4f}    "
      f"delta {100 * (s2['score'] - s1['score']):+.2f}pp")

# per-problem verdicts
d1 = {}
for line in (BANK1 / "lcb_result.samples.jsonl").open(errors="ignore"):
    if line.strip():
        r = json.loads(line)
        d1[r.get("task_id")] = (bool(r.get("passed")), len(r.get("completion") or ""))

d2 = {}
f2 = BANK2 / "lcb_result.samples.jsonl"
if f2.is_file():
    for line in f2.open(errors="ignore"):
        if line.strip():
            r = json.loads(line)
            d2[r.get("task_id")] = (bool(r.get("passed")), len(r.get("completion") or ""))
    src = "samples.jsonl"
else:
    pat = re.compile(r"\[(\d+)/77\]\s+(\S+)\s+(PASS|FAIL)\s+([\d.]+)s\s+chars=(\d+)")
    for line in LOG.open(errors="ignore"):
        m = pat.search(line)
        if m:
            d2[m.group(2)] = (m.group(3) == "PASS", int(m.group(5)))
    src = "log"

# cross-check the per-problem source against the canonical score
for name, d, s in (("draw1", d1, s1), ("draw2", d2, s2)):
    if d:
        got = sum(v[0] for v in d.values()) / len(d)
        if abs(got - s["score"]) > 1e-6:
            print(f"  WARNING {name}: per-problem rate {got:.4f} != canonical "
                  f"{s['score']:.4f} — pairing may not describe the banked cell")

common = [t for t in d2 if t in d1]
p1 = sum(d1[t][0] for t in common)
p2 = sum(d2[t][0] for t in common)
b = sum(1 for t in common if d1[t][0] and not d2[t][0])
c = sum(1 for t in common if d2[t][0] and not d1[t][0])
n = len(common)
p = mcnemar(b, c)

print(f"\n=== paired, SAME model, SAME basis, n={n} (verdicts from {src}) ===")
print(f"  draw1 {p1}/{n} = {p1/n:.4f}")
print(f"  draw2 {p2}/{n} = {p2/n:.4f}   delta {100*(p2-p1)/n:+.2f}pp")
print(f"  discordant: draw1-only={b}  draw2-only={c}  flips={b+c}  McNemar p={p:.4f}")
ch1 = sorted(d1[t][1] for t in common)
ch2 = sorted(d2[t][1] for t in common)
print(f"  chars p50: draw1 {ch1[len(ch1)//2]}  draw2 {ch2[len(ch2)//2]}")

print("\n=== the question this run was launched to answer ===")
print("  gepo2-vs-armJ (DIFFERENT models) was 13 flips, -9.33pp, p=0.1185")
print(f"  gepo2-vs-gepo2 (SAME model)     is {b+c} flips, {100*(p2-p1)/n:+.2f}pp, p={p:.4f}")
verdict = ("WITHIN the noise floor — the LCB reversal is draw noise"
           if abs(p2 - p1) / n >= 0.0933 - 1e-9 or (b + c) >= 13
           else "BELOW the between-model gap — floor is narrower than the effect")
print(f"  -> {verdict}")
