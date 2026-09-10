#!/usr/bin/env python3
"""Offline counterfactual re-scoring of GRPO rollouts under alternative penalty shapes.

WHY THIS EXISTS
---------------
`grpo_an_short` (2026-09-10, Jprime-p3, 300 steps, 7h59m) returned a NULL:
completion length t=-0.41, reward t=+0.39. The cause was not the learning rate --
it was that a large share of prompt GROUPS carried no within-group reward
contrast, and GRPO's advantage is computed strictly within a group. A group whose
rollouts all score identically contributes exactly zero gradient no matter how
long you train.

The reward fn already dumped one row per rollout carrying BOTH the unclamped
ratio `raw = ntok/budget` and the clamped penalty `pen` that was actually
applied. That makes the counterfactual exact:

    r_counterfactual = r + lam*pen - lam*f(raw)

So a different penalty shape can be evaluated on the SAME 9600 rollouts with no
GPU and no second training run. This script does that and reports, per tier, the
metric that actually decides whether a run can learn: the share of groups with
non-zero reward variance.

USAGE
    python rescore_grpo_penalty_shapes.py <rollouts_*.jsonl> [...]
"""
import json
import math
import statistics as st
import sys
from collections import defaultdict

# Penalty shapes. Each maps raw=ntok/budget -> penalty in roughly [0, 1+].
# `clamped` reproduces what the null run actually applied, and is the control:
# if it does not reproduce the observed dead-group rate, the harness is wrong and
# every other row is void.
SHAPES = {
    "clamped (RUN)": lambda x: min(x, 1.0),
    "unclamped":     lambda x: x,
    "log1p":         lambda x: math.log1p(x) / math.log(2.0),
    "sqrt":          lambda x: math.sqrt(x),
    "clamp@1.5":     lambda x: min(x, 1.5),
    "clamp@2.0":     lambda x: min(x, 2.0),
    "tanh":          lambda x: math.tanh(x),
}


def load(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
    return rows


def regroup(rows):
    """Group key must include rank: gid restarts per rank, so gid alone COLLIDES
    two different groups into one and fabricates variance that never existed."""
    g = defaultdict(list)
    for r in rows:
        g[(r.get("rank", 0), r["gid"])].append(r)
    return g


def score(groups, f):
    """Per-tier: share of groups with non-zero reward std, and mean std."""
    per = defaultdict(lambda: {"n": 0, "alive": 0, "stds": []})
    for _, grp in groups.items():
        if len(grp) < 2:
            continue
        tier = grp[0]["tier"]
        vals = []
        for r in grp:
            lam = float(r.get("lam") or 0.0)
            raw = r.get("raw")
            pen = r.get("pen")
            base = r["r"] + lam * (pen or 0.0)
            vals.append(base - lam * f(raw) if raw is not None else r["r"])
        s = st.pstdev(vals)
        d = per[tier]
        d["n"] += 1
        d["stds"].append(s)
        if s > 1e-9:
            d["alive"] += 1
    return per


def main(paths):
    rows = load(paths)
    groups = regroup(rows)
    print(f"rollouts={len(rows)}  groups={len(groups)}  "
          f"(grouped by (rank,gid) -- gid alone would collide ranks)\n")

    tiers = sorted({r["tier"] for r in rows})
    w = max(len(t) for t in tiers) + 2
    hdr = "%-16s" % "shape" + "".join("%*s" % (w + 16, t) for t in tiers)
    print(hdr)
    print("%-16s" % "" + "".join("%*s" % (w + 16, "alive%   mean_std") for _ in tiers))
    print("-" * len(hdr))
    base_alive = {}
    for name, f in SHAPES.items():
        per = score(groups, f)
        cells = ""
        for t in tiers:
            d = per.get(t)
            if not d or not d["n"]:
                cells += "%*s" % (w + 16, "-")
                continue
            alive = d["alive"] / d["n"]
            if name.startswith("clamped"):
                base_alive[t] = alive
            cells += "%*s" % (w + 16, "%6.1f%%   %7.4f" % (100 * alive, st.mean(d["stds"])))
        print("%-16s%s" % (name, cells))

    print("\nDELTA in live-group share vs the shape the null run actually used:")
    for name, f in SHAPES.items():
        if name.startswith("clamped ("):
            continue
        per = score(groups, f)
        parts = []
        for t in tiers:
            d = per.get(t)
            if not d or not d["n"]:
                continue
            parts.append("%s %+5.1fpp" % (t, 100 * (d["alive"] / d["n"] - base_alive[t])))
        print("  %-14s %s" % (name, "   ".join(parts)))

    print("\nWHY SOME GROUPS CANNOT BE REVIVED BY ANY PENALTY SHAPE:")
    for t in tiers:
        gs = [g for g in groups.values() if g and g[0]["tier"] == t and len(g) >= 2]
        lt2 = sum(1 for g in gs if sum(1 for r in g if r["ok"]) < 2)
        allover = sum(1 for g in gs
                      if sum(1 for r in g if r["ok"]) >= 2
                      and all((r.get("raw") or 0) > 1.0 for r in g if r["ok"]))
        print("  %-14s groups=%4d  <2 passers=%5.1f%% (PASS-RATE problem, not the clamp)"
              "   all-passers-over-budget=%5.1f%% (CLAMP problem)"
              % (t, len(gs), 100 * lt2 / max(len(gs), 1), 100 * allover / max(len(gs), 1)))

    sweep_budget(groups, tiers)
    direction(groups, tiers)


def direction(groups, tiers):
    """Is the within-group variance pointing TOWARD SHORTER?

    WHY THIS EXISTS: the budget sweep shows live-group share and within-group std rise
    monotonically as the budget is loosened -- but at x2.0, 91-100% of rollouts land
    UNDER budget. Variance is maximal there precisely BECAUSE the budget has stopped
    being a target: nothing is being asked to get under a bar any more. So "there is
    within-group variance" is NOT sufficient evidence that the run can learn brevity,
    and a pilot gated on std alone can pass while length refuses to move -- which is
    exactly the null we just paid 8 GPU-hours for.

    The metric that does discriminate is the SIGN: within a group, does reward fall as
    length rises? Spearman(ntok, reward) < 0 means the gradient points toward shorter.
    A group with large std but ~0 correlation contributes noise, not brevity pressure.
    """
    print("\nGRADIENT DIRECTION -- within-group Spearman(ntok, reward), by budget multiplier")
    print("  mean_rho < 0 => reward falls as length rises => pressure toward SHORTER")
    print("  %neg = share of live groups whose rho < 0.  std WITHOUT sign is not signal.")
    hdr = "%-10s" % "mult" + "".join("%26s" % t for t in tiers)
    print(hdr)
    print("%-10s" % "" + "".join("%26s" % "mean_rho   %neg   n_live" for _ in tiers))
    for m in (0.8, 1.0, 1.25, 1.5, 2.0):
        f = lambda x, m=m: min(x / m, 1.0)
        per = {t: [] for t in tiers}
        for _, grp in groups.items():
            if len(grp) < 2:
                continue
            t = grp[0]["tier"]
            vals, toks = [], []
            for r in grp:
                lam = float(r.get("lam") or 0.0)
                raw, pen = r.get("raw"), r.get("pen")
                base = r["r"] + lam * (pen or 0.0)
                vals.append(base - lam * f(raw) if raw is not None else r["r"])
                toks.append(r["ntok"])
            if st.pstdev(vals) <= 1e-9:
                continue
            per[t].append(_spearman(toks, vals))
        cells = ""
        for t in tiers:
            v = [x for x in per[t] if x is not None]
            cells += "%26s" % ("-" if not v else "%+8.3f  %5.1f%%  %5d" % (
                st.mean(v), 100 * sum(1 for x in v if x < 0) / len(v), len(v)))
        print("%-10s%s" % ("x%.2f" % m, cells))


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _spearman(a, b):
    """Spearman rho. Returns None when either side is constant (rho undefined) --
    a constant side must not be silently reported as rho=0, which would read as
    'no direction' when the truth is 'no measurement'."""
    ra, rb = _rank(a), _rank(b)
    if st.pstdev(ra) == 0 or st.pstdev(rb) == 0:
        return None
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def sweep_budget(groups, tiers):
    """Budget placement sweep, on the same rollouts.

    The module docstring of grpo_reward_efficiency.py states the design rule: rollouts
    UNDER budget sit on the sloped part and are differentiated; rollouts OVER budget all
    clip to the same penalty and are not. So WHERE the budget sits relative to the
    group's length distribution is the whole design -- AN's working operating point had
    budget ~0.66x the median length.

    Because every dumped row carries raw = ntok/budget, a budget multiplier m is exactly
    raw' = raw/m. That makes budget placement sweepable offline on the SAME rollouts,
    with no GPU -- the same trick as the penalty shape sweep.
    """
    print("\nBUDGET PLACEMENT SWEEP (clamped shape held fixed, only the budget moves)")
    print("  m>1 = looser budget (more rollouts land under it, more get graded)")
    hdr = "%-10s" % "mult" + "".join("%24s" % t for t in tiers)
    print(hdr)
    print("%-10s" % "" + "".join("%24s" % "alive%   mean_std   %under" for _ in tiers))
    for m in (0.6, 0.8, 1.0, 1.25, 1.5, 2.0):
        f = lambda x, m=m: min(x / m, 1.0)
        per = score(groups, f)
        cells = ""
        for t in tiers:
            d = per.get(t)
            if not d or not d["n"]:
                cells += "%24s" % "-"
                continue
            gs = [g for g in groups.values() if g and g[0]["tier"] == t and len(g) >= 2]
            ok = [r for g in gs for r in g if r["ok"] and r.get("raw") is not None]
            und = sum(1 for r in ok if r["raw"] / m < 1.0) / max(len(ok), 1)
            cells += "%24s" % ("%6.1f%%   %7.4f   %5.1f%%" % (
                100 * d["alive"] / d["n"], st.mean(d["stds"]), 100 * und))
        print("%-10s%s" % ("x%.2f" % m, cells))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
