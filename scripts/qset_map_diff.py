"""Compare two competence maps built from DISJOINT 1-trace-per-language q-sets.

THE QUESTION: how many traces per language does the targeting map need? If two
independent 1q maps rank experts the same, 1q is enough. If they move marginally, 2q.
If they diverge, keep adding sets until the movement flattens.

WHAT IS COMPARED, and why not just the raw scores. The map's raw per-expert numbers are
not the deliverable -- the DROP MAP is. Two maps can differ numerically everywhere and
still drop exactly the same experts, which would mean 1q is sufficient. So this reports
three things, cheapest-to-decide last:

  1. per-category rank agreement  (Spearman + top-K overlap) -- is the ORDERING stable?
  2. drop-set symmetric difference at the real drop-count -- does the SHIPPED map change?
  3. the experts that flip, per layer -- so a small number can be eyeballed.

Only `targeted_*` categories are compared. The `generic_*` categories are imported
identically from the same Tier-A source in both runs, so including them would dilute
every statistic toward "identical" and hide the Tier-B movement this exists to measure.
"""
import argparse
import json
from collections import defaultdict


def load(path):
    d = json.load(open(path))
    return d["categories"], d.get("metadata", {})


def score_of(e, kind):
    if kind == "tc":
        return float(e.get("tc", 0.0))
    if kind == "wnorm":
        return float(e.get("wnorm", 0.0))
    if kind == "wnorm_tc":
        return float(e.get("wnorm", 0.0)) * float(e.get("tc", 0.0))
    raise ValueError(kind)


def spearman(a, b):
    """Rank correlation without scipy. a, b are score lists over the same experts."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):           # average ties, else ties inflate rho
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


ap = argparse.ArgumentParser()
ap.add_argument("map_a")
ap.add_argument("map_b")
ap.add_argument("--score", default="tc", choices=["tc", "wnorm", "wnorm_tc"])
ap.add_argument("--only-cats", default="",
                help="comma-separated targeted category suffixes to restrict to. "
                     "A convergence curve must be read on ONE basis: the 3q point "
                     "only exists for the languages with >=6 PASS traces, so the "
                     "1q and 2q points have to be recomputed over those same "
                     "languages or the curve compares different language mixes.")
ap.add_argument("--drop-count", type=int, default=30,
                help="experts dropped per layer; the symmetric difference is measured "
                     "at this boundary because that is what actually ships")
ap.add_argument("--topk", type=int, default=30)
args = ap.parse_args()

A, metaA = load(args.map_a)
B, metaB = load(args.map_b)
cats = sorted(c for c in A if c.startswith("targeted_") and c in B)
if args.only_cats:
    want = {("targeted_" + x) if not x.startswith("targeted_") else x
            for x in args.only_cats.split(",") if x.strip()}
    missing = sorted(want - set(cats))
    if missing:
        raise SystemExit("FATAL: --only-cats named %s, absent from one of the maps" % missing)
    cats = [c for c in cats if c in want]
skipped = sorted(set(A) ^ set(B))
print("=== %s  vs  %s ===" % (args.map_a.split("/")[-1], args.map_b.split("/")[-1]))
print("  score=%s  drop_count=%d  topK=%d" % (args.score, args.drop_count, args.topk))
print("  targeted categories in both: %d   %s" % (len(cats), cats))
if skipped:
    print("  NOTE categories not in both (ignored): %s" % skipped)
print()

print("=== 1. per-category rank agreement (over all layers) ===")
print("  %-26s %8s %10s %12s" % ("category", "rho", "topK ovl", "layers"))
overall = []
for c in cats:
    rhos, ovls = [], []
    for li in sorted(A[c], key=int):
        ea = {e["id"]: e for e in A[c][li]}
        eb = {e["id"]: e for e in B[c][li]}
        ids = sorted(set(ea) & set(eb))
        va = [score_of(ea[i], args.score) for i in ids]
        vb = [score_of(eb[i], args.score) for i in ids]
        rhos.append(spearman(va, vb))
        ta = {i for i in sorted(ids, key=lambda i: -score_of(ea[i], args.score))[:args.topk]}
        tb = {i for i in sorted(ids, key=lambda i: -score_of(eb[i], args.score))[:args.topk]}
        ovls.append(len(ta & tb) / max(len(ta), 1))
    r = sum(rhos) / len(rhos)
    o = sum(ovls) / len(ovls)
    overall.append((r, o))
    print("  %-26s %8.4f %9.1f%% %12d" % (c, r, 100 * o, len(rhos)))
print("  %-26s %8.4f %9.1f%%" % ("MEAN",
      sum(x for x, _ in overall) / len(overall), 100 * sum(y for _, y in overall) / len(overall)))
print()

print("=== 2. drop-set symmetric difference at drop_count=%d ===" % args.drop_count)
print("  (the shipped artifact — two maps that differ numerically can still drop the same experts)")
print("  %-26s %10s %10s %10s" % ("category", "layers", "diff/layer", "identical"))
flips = defaultdict(list)
for c in cats:
    diffs, same = [], 0
    for li in sorted(A[c], key=int):
        ea = {e["id"]: score_of(e, args.score) for e in A[c][li]}
        eb = {e["id"]: score_of(e, args.score) for e in B[c][li]}
        ids = sorted(set(ea) & set(eb))
        da = set(sorted(ids, key=lambda i: (ea[i], i))[:args.drop_count])
        db = set(sorted(ids, key=lambda i: (eb[i], i))[:args.drop_count])
        d = len(da ^ db) // 2
        diffs.append(d)
        same += (d == 0)
        if d:
            flips[c].append((li, d, sorted(da - db)[:6], sorted(db - da)[:6]))
    print("  %-26s %10d %10.2f %9d/%d" % (c, len(diffs), sum(diffs) / len(diffs), same, len(diffs)))
tot = sum(d for c in flips for _, d, _, _ in flips[c])
print("  TOTAL experts differing across all categories/layers: %d" % tot)
print()

print("=== 3. verdict ===")
mr = sum(x for x, _ in overall) / len(overall)
if tot == 0:
    print("  IDENTICAL drop sets — 1 trace/language is sufficient at this drop-count.")
elif mr > 0.95 and tot <= 2 * len(cats):
    print("  MARGINAL movement (rho=%.4f, %d experts) — 2 traces/language looks sufficient." % (mr, tot))
else:
    print("  SUBSTANTIAL movement (rho=%.4f, %d experts) — add another q-set and re-measure." % (mr, tot))
print("  NOTE: this compares two 1q samples. It bounds SAMPLING noise at 1q; it does not")
print("        prove a 2q map equals the population map. The converged answer is the q at")
print("        which successive sets stop moving the drop set.")
