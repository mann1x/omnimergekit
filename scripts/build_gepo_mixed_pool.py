#!/usr/bin/env python3
"""Build the REBALANCED GEPO pool: LCB brevity target + a large mixed replay body.

WHY REBALANCE -- THE E2B CONTROL
--------------------------------
GEPO on Gemma-4 E2B worked: mean_length 779.8 -> 683.9 (-12.3%) in one epoch with no
score damage. R9 on CoderX produced comparable brevity (run2 6593 -> 6031, -8.5%; the
eval measured LCB tok -12.7% for gepo2) and yet LOST capability: GPQA -7.58pp,
HumanEval -3.66pp. The two runs differ in ways that all push the same direction:

    knob             AN E2B (worked)            R9 run2 (hurt scores)
    pool             894 problems, 3 sources    128 problems, LCB only
                     gsm8k 400 / math 294 /
                     aime 200
    length-budget    512  (BELOW mean 780)      12288 (mean was 6593 -> ratio 0.54)
    max-completion   1024                       12288
    learning rate    1e-6                       5e-6   (5x)
    temperature      0.9                        0.6

A 5x learning rate on a 7x smaller SINGLE-DOMAIN pool is the most economical
explanation of what was actually measured -- scores fell while completion length stayed
FLAT on the damaged benches (GPQA 804->792, HE 220->223). That is drift from
overfitting a narrow domain, not brevity pressure eroding reasoning.
[[project_gpqa_cot_needs_16k_on_armj]]

So this pool stops being "LCB plus a small counterweight" and becomes a broad mixture,
in AN's spirit: the brevity target is a MINORITY of the pool, and most rows are cheap,
diverse, correctness-gated problems that hold capability.

THINKING AND NO-THINKING, BOTH, PER ROW
---------------------------------------
An all-thinking pool does not work: GPQA with thinking on runs 16-18k tokens, 2 of 4
rollouts never reach an answer inside 24576, and the tier scores zero for every
rollout. An all-no-think pool is also wrong: it would train the model out of thinking
entirely. So `meta.think` is a PER-ROW property here, and the trainer renders each row
accordingly:

    lcb_exec     think=True   -- the brevity target keeps its reasoning
    mbpp_exec    both         -- ~3400 tok with thinking, ~205 without; affordable
    mc_letter    think=False  -- 16-18k with thinking is not affordable at any G=8
                                 budget that fits 96 GB, and half never answer anyway

COST IS THE BINDING CONSTRAINT, NOT PROBLEM COUNT
-------------------------------------------------
AN's 894 problems were CHEAP: ~780 tokens each. Ours run to 6000. Matching AN's count
is not matching AN's compute. Measured tokens per problem at G=8:

    lcb_exec (think)      ~48000        mc_letter (no-think)  ~12800
    mbpp_exec (think)     ~27000        mbpp_exec (no-think)   ~1600

against run2's realised ~307k tok/h. The default composition below is sized to land
near 30 h while putting ~88% of the ROWS outside LCB.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
# Measured tokens per rollout, G=8. Used only to PRICE a composition up front, so an
# unaffordable pool is rejected before the run rather than discovered 20 h in.
TOK = {("lcb_exec", True): 6000, ("mc_letter", False): 1595,
       ("mbpp_exec", True): 3400, ("mbpp_exec", False): 205,
       # The efficiency tier is mc_letter, so no-think reuses that measured figure.
       # The THINKING figure is a placeholder and is flagged as such at build time:
       # these items are far shorter than GPQA, but nobody has measured them, and a
       # pool priced on a guess is how a run discovers it is unaffordable 20 h in.
       ("efficiency", False): 1595, ("efficiency", True): 2500}
# A tok/h rate DOES NOT TRANSFER BETWEEN POOLS, and this constant used to pretend it
# did. r9_gepo_run4.sh measured both and says so plainly:
#
#     run2  1024 rollouts, mean_tok 5780, LCB-only  -> 296k tok/h
#     run4  6792 rollouts, mean_tok ~1860, 4 tiers  -> 187k tok/h  (at step 77)
#
# The old value here was 307k, which is neither of those. Priced with it, the 849-row
# pool that actually shipped for run4 came out at 41.2 h against a 32 h budget and this
# script REFUSED to build it -- the guard rejecting the one composition known to work.
# A guard that refuses correct configurations teaches people to pass --hours-budget and
# stop reading it, which is worse than no guard.
#
# So the rate is now a flag, defaulted to the mixed-pool measurement rather than the
# LCB-only one, and the docstring above no longer claims a single number is portable.
# Take elapsed/steps from a run's own first ~20 steps and pass --rate; never tqdm's s/it.
DEFAULT_RATE = 187_000  # tokens/hour, run4 mixed pool measured at step 77


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcb-pool", default=str(REPO / "eval/lcb/lcb_rl_pool.jsonl"))
    ap.add_argument("--replay-pool",
                    default=str(REPO / "eval/replay/gepo_replay_pool.jsonl"))
    ap.add_argument("--out", default=str(REPO / "eval/replay/gepo_mixed_pool.jsonl"))
    ap.add_argument("--lcb", type=int, default=128)
    ap.add_argument("--gpqa-nothink", type=int, default=250)
    ap.add_argument("--mbpp-nothink", type=int, default=371)
    ap.add_argument("--mbpp-think", type=int, default=100)
    ap.add_argument("--efficiency-pool",
                    default=str(REPO / "eval/efficiency/gepo_efficiency_pool.jsonl"))
    ap.add_argument("--efficiency", type=int, default=0,
                    help="Rows from the efficiency tier (troubleshooting decision "
                         "policy, mined from the manic harness arm contrast). Default "
                         "0: it is opt-in because the tier is small, and a small tier "
                         "repeated to fill a quota is correlated rows, not more data.")
    ap.add_argument("-G", "--group", type=int, default=8)
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE,
                    help="Generation rate in tokens/hour for the hours estimate. "
                         "Measure it on the pool you are actually running.")
    ap.add_argument("--hours-budget", type=float, default=72.0,
                    help="Refuse above this. Default admits the 849-row run4 pool "
                         "(67.7 h at the measured mixed-pool rate), which the previous "
                         "default of 32 h rejected.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    def load(p):
        f = pathlib.Path(p)
        if not f.is_file():
            sys.exit(f"REFUSE: missing pool {f}")
        return [json.loads(x) for x in f.open() if x.strip()]

    lcb, rep = load(a.lcb_pool), load(a.replay_pool)
    for r in lcb:
        r.setdefault("meta", {}).setdefault("reward_kind", "lcb_exec")
    by: dict[str, list] = {}
    for r in rep:
        by.setdefault(r["meta"]["reward_kind"], []).append(r)

    rng = random.Random(a.seed)
    for v in by.values():
        rng.shuffle(v)
    rng.shuffle(lcb)

    eff = []
    if a.efficiency:
        ep = pathlib.Path(a.efficiency_pool)
        if not ep.exists():
            sys.exit(f"REFUSE: --efficiency {a.efficiency} but no pool at {ep}. "
                     "Build it with build_gepo_efficiency_pool.py first.")
        eff = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
        if a.efficiency > len(eff):
            sys.exit(f"REFUSE: asked for {a.efficiency} efficiency rows but the pool "
                     f"holds {len(eff)}. Sampling with replacement would put the same "
                     "prompt in one epoch twice and silently double its weight; write "
                     "more items instead.")
        rng.shuffle(eff)

    want = [("lcb_exec", True, a.lcb, lcb),
            ("mc_letter", False, a.gpqa_nothink, by.get("mc_letter", [])),
            ("mbpp_exec", False, a.mbpp_nothink, by.get("mbpp_exec", [])),
            ("mbpp_exec", True, a.mbpp_think, by.get("mbpp_exec", [])),
            # Its rows already carry reward_kind mc_letter; the label here is only for
            # the cost table and the composition print, so the tier is visible rather
            # than hidden inside the GPQA count.
            ("efficiency", None, a.efficiency, eff)]

    # mbpp think/no-think must be DISJOINT problems. The same problem in both modes
    # would put two correlated rows in one epoch and quietly double its weight.
    used_mbpp: set[str] = set()
    rows, cost = [], 0.0
    for kind, think, n, src in want:
        if n <= 0:
            continue
        pool = [r for r in src if r["id"] not in used_mbpp] if kind == "mbpp_exec" else src
        if len(pool) < n:
            sys.exit(f"REFUSE: asked for {n} {kind}(think={think}) rows but only "
                     f"{len(pool)} are available (disjoint). Lower the count.")
        for r in pool[:n]:
            if kind == "mbpp_exec":
                used_mbpp.add(r["id"])
            m = dict(r["meta"])
            # The efficiency tier keeps the reward_kind and think flag its own builder
            # chose. Overwriting them here would silently re-price a tier that was
            # written deliberately, and reward_kind must stay mc_letter or the
            # dispatch has no verifier for it.
            if kind != "efficiency":
                m["reward_kind"] = kind
                m["think"] = think
            m.setdefault("length_lambda", 0.7 if kind == "lcb_exec" else 0.0)
            rows.append({"id": f"{r['id']}#{'T' if think else 'N'}",
                         "source": r.get("source", kind),
                         "prompt": r["prompt"], "gold": str(r.get("gold") or ""),
                         "meta": m})
        # The efficiency tier carries its own think flag per row, so the cost is
        # looked up from what the rows actually say rather than from a tuple field
        # this tier deliberately leaves unset.
        think_for_cost = (bool(pool[0]["meta"].get("think", True))
                          if kind == "efficiency" else think)
        cost += n * a.group * TOK[(kind, think_for_cost)]
        if kind == "efficiency" and think_for_cost:
            print("NOTE: the efficiency tier is priced at a PLACEHOLDER 2,500 tok with "
                  "thinking on -- that figure is not measured. Measure the tier's real "
                  "completion length before trusting the hours estimate below.")

    rng.shuffle(rows)
    hours = cost / a.rate
    # DRIVER vs REPLAY, not lcb vs everything-else. A driver is any tier under length
    # pressure -- the thing the run is trying to move. Everything else is there to hold
    # capability while it moves. Reporting the efficiency tier as "replay" because it is
    # not LCB is how a pool comes to be 5% driver without anyone noticing: run4 carried
    # 128 driver rows in 849 and its pool-wide mean_length moved +1.3% over 93 rounds,
    # a number that says nothing either way about the tier it was meant to measure.
    n_driver = sum(1 for r in rows if float(r["meta"].get("length_lambda", 0)) > 0)
    n_lcb = sum(1 for r in rows if r["meta"]["reward_kind"] == "lcb_exec")
    n_think = sum(1 for r in rows if r["meta"]["think"])
    share = 100 * n_driver / len(rows)
    print(f"rows: {len(rows)}  (lcb {n_lcb} = {100*n_lcb/len(rows):.0f}%)")
    print(f"DRIVER (length_lambda > 0): {n_driver} = {share:.0f}%  |  "
          f"replay (correctness-only): {len(rows)-n_driver} = {100-share:.0f}%")
    if 0 < share < 25:
        print(f"  WARNING: a {share:.0f}% driver is easy to dilute past the point of "
              "measurement. run4 ran at 15% and could not tell whether its driver tier "
              "moved at all. Read the PER-TIER numbers, never the pool-wide mean.")
    if n_driver == 0:
        print("  WARNING: no row carries length_lambda > 0, so nothing in this pool is "
              "under length pressure. It will hold capability and move nothing.")
    print(f"thinking rows: {n_think} = {100*n_think/len(rows):.0f}%  |  "
          f"no-think: {len(rows)-n_think}")
    comp: dict[str, int] = {}
    for r in rows:
        comp[f"{r['meta']['reward_kind']}/{'think' if r['meta']['think'] else 'nothink'}"] = \
            comp.get(f"{r['meta']['reward_kind']}/{'think' if r['meta']['think'] else 'nothink'}", 0) + 1
    print("composition:", dict(sorted(comp.items())))
    print(f"PRICED at G={a.group}: {cost/1e6:.2f}M tokens -> ~{hours:.1f} h "
          f"at {a.rate/1000:.0f}k tok/h")
    if hours > a.hours_budget:
        sys.exit(f"REFUSE: {hours:.1f} h exceeds --hours-budget {a.hours_budget}. "
                 "An unaffordable pool must be rejected here, not discovered 20 h into "
                 "the run. Lower --lcb (it dominates: 48k tok/problem) or --mbpp-think.")
    pathlib.Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
