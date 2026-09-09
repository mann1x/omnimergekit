#!/usr/bin/env python3
"""How much length signal does `lenpen = min(nt/budget, 1.0)` actually destroy?

THE QUESTION
------------
The efficiency reward clamps the length penalty at the budget, so every passing rollout
at or above budget receives the IDENTICAL penalty lambda. Ordering above the budget is
discarded: a 1,300-token answer and a 3,000-token answer score the same. With
BUDGET_QUANTILE = 0.35 roughly two thirds of passers land in that flat zone by
construction.

It does NOT follow that two thirds of the gradient is gone. GRPO standardises rewards
WITHIN a group, so a group still carries length information as long as its passers
straddle the budget -- the surviving signal is binary (under vs over) instead of graded.
A group is length-dead only when EVERY passer is above budget.

This script answers the question by measurement, not by assumption, and it does it
offline: `--dump-rollouts` records nt/budget both clamped and unclamped, so the same
rollouts can be re-scored under any penalty shape with no regeneration and no GPU.

WHAT IT COMPUTES, per tier and overall
--------------------------------------
  groups                 total groups with >= 2 rollouts
  lt2pass                groups with < 2 passers -- dead for a PASS-RATE reason, not the
                         clamp. Counted separately: merging the two blames the clamp for
                         a problem it did not cause.
  clamp_dead             groups with >= 2 passers, ALL at/above budget. THIS is the
                         clamp's cost: the length term contributes exactly zero variance.
  revived                clamp_dead groups that regain nonzero length variance once the
                         clamp is lifted. The headline number.
  |corr(ntok, adv)|      mean |Pearson| between completion length and the standardised
                         advantage among passers, clamped vs unclamped. How much the
                         gradient actually tracks length.
  sd(adv) among passers  the spread the length term produces, clamped vs unclamped.

The counterfactual reward is exact, not modelled:  r_unclamped = r + lam*pen - lam*raw
where `pen` is the clamped penalty that was applied and `raw` = nt/budget.

READ THE VERDICT WITH THE PASS RATE IN HAND. A tier whose groups are mostly `lt2pass`
(lcb_exec/T at pass 0.196) is telling you about its pass rate; the clamp cannot be
blamed for, or fixed by, anything there.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
from collections import defaultdict


def standardise(vals: list[float]) -> list[float]:
    """GRPO's within-group advantage: centre, then scale by the group sd."""
    if len(vals) < 2:
        return [0.0] * len(vals)
    mu = st.mean(vals)
    sd = st.pstdev(vals)
    if sd <= 0:
        return [0.0] * len(vals)
    return [(v - mu) / sd for v in vals]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_glob",
                    help="e.g. /mnt/sdc/v7rework/rollouts.rank*.jsonl")
    ap.add_argument("--ceiling", type=float, default=None,
                    help="Counterfactual clamp ceiling instead of removing it entirely "
                         "(e.g. 3.0). Default: no clamp at all.")
    a = ap.parse_args()

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted(glob.glob(a.dump_glob)):
        with open(path) as fh:
            for ln in fh:
                r = json.loads(ln)
                groups[(r["rank"], r["gid"])].append(r)
    if not groups:
        print(f"REFUSE: no rollouts matched {a.dump_glob!r}. A silent empty analysis "
              "would print a clean-looking zero for every column.")
        return 2

    def cf(row: dict) -> float:
        """Reward with the clamp lifted (or raised to --ceiling). Exact, not modelled."""
        raw = row["raw"]
        if a.ceiling is not None:
            raw = min(raw, a.ceiling)
        return row["r"] + row["lam"] * row["pen"] - row["lam"] * raw

    per = defaultdict(lambda: {"groups": 0, "lt2pass": 0, "clamp_dead": 0, "revived": 0,
                               "corr_c": [], "corr_u": [], "sd_c": [], "sd_u": []})
    for rows in groups.values():
        if len(rows) < 2:
            continue
        tier = rows[0]["tier"]
        d = per[tier]
        d["groups"] += 1
        adv_c = standardise([r["r"] for r in rows])
        adv_u = standardise([cf(r) for r in rows])
        pidx = [i for i, r in enumerate(rows) if r["ok"]]
        if len(pidx) < 2:
            d["lt2pass"] += 1
            continue
        lens = [float(rows[i]["ntok"]) for i in pidx]
        pc = [adv_c[i] for i in pidx]
        pu = [adv_u[i] for i in pidx]
        sd_c, sd_u = st.pstdev(pc), st.pstdev(pu)
        d["sd_c"].append(sd_c)
        d["sd_u"].append(sd_u)
        if sd_c <= 1e-12:                       # every passer tied at the clamp
            d["clamp_dead"] += 1
            if sd_u > 1e-12:
                d["revived"] += 1
        cc, cu = pearson(lens, pc), pearson(lens, pu)
        if cc is not None:
            d["corr_c"].append(abs(cc))
        if cu is not None:
            d["corr_u"].append(abs(cu))

    lab = "no clamp" if a.ceiling is None else f"ceiling {a.ceiling:g}"
    print(f"counterfactual: {lab}\n")
    hdr = (f"{'tier':16s} {'groups':>7s} {'lt2pass':>8s} {'clamp_dead':>11s} "
           f"{'revived':>8s} {'|corr| clamp':>13s} {'|corr| cf':>10s} "
           f"{'sd clamp':>9s} {'sd cf':>8s}")
    print(hdr)
    print("-" * len(hdr))
    tot = defaultdict(float)
    for tier, d in sorted(per.items()):
        m = lambda v: (sum(v) / len(v)) if v else float("nan")  # noqa: E731
        print(f"{tier:16s} {d['groups']:7d} {d['lt2pass']:8d} {d['clamp_dead']:11d} "
              f"{d['revived']:8d} {m(d['corr_c']):13.3f} {m(d['corr_u']):10.3f} "
              f"{m(d['sd_c']):9.3f} {m(d['sd_u']):8.3f}")
        for k in ("groups", "lt2pass", "clamp_dead", "revived"):
            tot[k] += d[k]
    g = tot["groups"] or 1
    print(f"\nOVERALL: {tot['groups']:.0f} groups | "
          f"clamp-dead {tot['clamp_dead']:.0f} ({tot['clamp_dead'] / g:.1%}) | "
          f"revived by lifting the clamp {tot['revived']:.0f} "
          f"({tot['revived'] / g:.1%}) | "
          f"dead for pass-rate reasons {tot['lt2pass']:.0f} "
          f"({tot['lt2pass'] / g:.1%}, NOT the clamp)")
    print("\nThe clamp is worth changing only if `revived` is a large share of groups AND "
          "|corr| rises materially. If `lt2pass` dominates, fix the pass rate instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
