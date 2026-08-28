"""Paired LCB comparison across all four arms, judged against the MEASURED noise floor.

The gepo2-vs-gepo2 repeat draw (2026-08-26) gave 20 discordant / +5.19pp / p=0.5034 on
this exact bench and basis -- a paired SE of sqrt(20)/77 = 5.8pp. Any delta below ~11-12pp
is inside two sigma of pure resampling. Deltas are therefore reported WITH their flip
counts and McNemar p, never alone.
"""
import json
import os
from math import comb

R = "/srv/ml/eval_results/qwen_suite/lcb_v6_77q"
ARMS = [("armJ", "qwenhybridp24_q6k"), ("gepo1", "qwena3bgepo1_q6k"),
        ("gepo2", "qwena3bgepo2_q6k"), ("gepo3", "qwena3bgepo3_q6k")]
FLOOR_SE = (20 ** 0.5) / 77 * 100


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(cell):
    d = {}
    p = os.path.join(R, cell, "lcb_result.samples.jsonl")
    if not os.path.isfile(p):
        return None
    for line in open(p, errors="ignore"):
        if line.strip():
            r = json.loads(line)
            d[r.get("task_id")] = (bool(r.get("passed")), len(r.get("completion") or ""))
    return d


data = {a: load(c) for a, c in ARMS}
missing = [a for a, d in data.items() if not d]
if missing:
    raise SystemExit("missing samples for: " + ", ".join(missing))

print(f"measured floor on THIS bench/basis: paired SE = {FLOOR_SE:.2f}pp "
      f"(from the 20-flip gepo2-vs-gepo2 repeat)")
print(f"=> |delta| < {2*FLOOR_SE:.1f}pp is inside 2 SE of pure resampling\n")

print("%-16s %7s %7s %9s %6s %6s %7s %9s  %s" %
      ("pair", "A", "B", "delta", "A-only", "B-only", "flips", "McNemar", "verdict"))
print("-" * 104)
for i, (na, _) in enumerate(ARMS):
    for nb, _ in ARMS[i + 1:]:
        A, B = data[na], data[nb]
        common = [t for t in A if t in B]
        n = len(common)
        pa = sum(A[t][0] for t in common)
        pb = sum(B[t][0] for t in common)
        b = sum(1 for t in common if A[t][0] and not B[t][0])
        c = sum(1 for t in common if B[t][0] and not A[t][0])
        d = (pb - pa) / n * 100
        p = mcnemar(b, c)
        sig = "RESOLVED" if abs(d) >= 2 * FLOOR_SE and p < 0.05 else \
              ("p<0.05 but inside floor" if p < 0.05 else "inside noise")
        print("%-16s %7.4f %7.4f %+9.2f %6d %6d %7d %9.4f  %s" %
              (f"{na}->{nb}", pa / n, pb / n, d, b, c, b + c, p, sig))

print("\n=== completion length on the SAME problems (chars) ===")
print("%-10s %9s %9s %9s   %s" % ("arm", "p50", "mean", "total", "vs armJ"))
base = None
for na, _ in ARMS:
    A = data[na]
    ch = sorted(v[1] for v in A.values())
    tot = sum(ch)
    if base is None:
        base = tot
    print("%-10s %9d %9d %9d   %+7.1f%%" %
          (na, ch[len(ch) // 2], tot / len(ch), tot, (tot - base) / base * 100))
