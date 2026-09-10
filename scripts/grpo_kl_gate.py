#!/usr/bin/env python3
"""KL-direction gate for a GRPO run. Answers ONE question: is the policy moving?

WHY KL DIRECTION AND NOT REWARD OR STD
--------------------------------------
`grpo_an_short` (2026-09-10) ran 300 steps / 7h59m to discover a null. Neither of the
obvious in-run signals would have caught it earlier:

  * REWARD has no power at pilot length -- the full 300-step run's own reward slope was
    t=+0.39. Gating 30 steps on reward gates on noise.
  * WITHIN-GROUP STD is not a health metric. Re-scoring that run's rollouts under a
    budget multiplier showed live-group share and std rise MONOTONICALLY as the budget
    is loosened, because at a loose budget ~all rollouts sit under it -- variance is
    maximal precisely when the budget has stopped being a target.

KL direction does have power, because it is a statement about the WEIGHTS rather than
about the reward. A policy genuinely moving away from its reference has RISING KL. In
the null run KL FELL, 0.0199 -> 0.0027: the beta anchor beat the reward gradient and
the policy sat still. That signature was visible within the first tens of steps and
would have ended the run in ~40 minutes instead of 8 hours.

ROBUSTNESS: the null run logged a single kl=39366.3 excursion at step 193 (loss=373)
whose grad_norm was only 0.129 -- a logp computation artefact that never reached the
weights. A mean-based test is destroyed by that one row, so this gate is RANK-BASED
(Spearman) and additionally reports a median split, which one outlier cannot move.

USAGE
    python grpo_kl_gate.py <run_dir_or_trainer_state.json> [--min-steps 25]
"""
import argparse
import json
import os
import statistics as st
import sys


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if st.pstdev(ra) == 0 or st.pstdev(rb) == 0:
        return None
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def find_state(p):
    if os.path.isfile(p):
        return p
    cands = []
    for d in os.listdir(p):
        if d.startswith("checkpoint-"):
            f = os.path.join(p, d, "trainer_state.json")
            if os.path.isfile(f):
                try:
                    cands.append((int(d.split("-")[1]), f))
                except ValueError:
                    continue
    if not cands:
        sys.exit(f"no trainer_state.json under {p}")
    return max(cands)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--min-steps", type=int, default=25)
    a = ap.parse_args()

    d = json.load(open(find_state(a.run)))
    h = [e for e in d["log_history"] if "kl" in e]
    if len(h) < a.min_steps:
        print(f"WAIT: only {len(h)} logged steps, gate needs {a.min_steps}")
        return 0

    steps = [e["step"] for e in h]
    kl = [e["kl"] for e in h]
    rho = spearman(steps, kl)
    n = len(kl)
    lo, hi = st.median(kl[: n // 3]), st.median(kl[-n // 3:])
    out = [x for x in kl if x > 100 * st.median(kl)]

    print(f"steps={n}  kl median={st.median(kl):.5f}")
    if out:
        print(f"  NOTE {len(out)} outlier row(s) >100x median (max {max(out):.1f}); "
              f"gate is rank-based so these do not move it")
    print(f"  Spearman(step, kl) = {rho:+.3f}" if rho is not None else "  Spearman: undefined")
    print(f"  median first third = {lo:.5f}   last third = {hi:.5f}   ratio = "
          f"{(hi / lo if lo else float('inf')):.2f}x")

    rising = (rho is not None and rho > 0) and hi > lo
    for tag, key in (("length", "completions/mean_length"), ("reward", "reward")):
        s = [(e["step"], e[key]) for e in d["log_history"] if key in e]
        if s:
            r2 = spearman([x for x, _ in s], [y for _, y in s])
            print(f"  (context) Spearman(step, {tag}) = "
                  + (f"{r2:+.3f}" if r2 is not None else "undefined"))

    print()
    if rising:
        print("GATE PASS: KL is rising -- the policy is moving. Continue.")
        return 0
    print("GATE FAIL: KL is FLAT or FALLING -- the policy is not moving away from its")
    print("  reference. This is the grpo_an_short signature. Raise --lr (or lower")
    print("  --beta) and restart; do NOT spend more steps hoping it turns around.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
