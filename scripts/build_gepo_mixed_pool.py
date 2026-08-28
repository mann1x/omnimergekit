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
       ("mbpp_exec", True): 3400, ("mbpp_exec", False): 205}
RATE = 307_000  # tokens/hour, from run2: 128 LCB problems, ~6.14M tokens, ~20 h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcb-pool", default=str(REPO / "eval/lcb/lcb_rl_pool.jsonl"))
    ap.add_argument("--replay-pool",
                    default=str(REPO / "eval/replay/gepo_replay_pool.jsonl"))
    ap.add_argument("--out", default=str(REPO / "eval/replay/gepo_mixed_pool.jsonl"))
    ap.add_argument("--lcb", type=int, default=96)
    ap.add_argument("--gpqa-nothink", type=int, default=250)
    ap.add_argument("--mbpp-nothink", type=int, default=300)
    ap.add_argument("--mbpp-think", type=int, default=150)
    ap.add_argument("-G", "--group", type=int, default=8)
    ap.add_argument("--hours-budget", type=float, default=32.0)
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

    want = [("lcb_exec", True, a.lcb, lcb),
            ("mc_letter", False, a.gpqa_nothink, by.get("mc_letter", [])),
            ("mbpp_exec", False, a.mbpp_nothink, by.get("mbpp_exec", [])),
            ("mbpp_exec", True, a.mbpp_think, by.get("mbpp_exec", []))]

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
            m["reward_kind"] = kind
            m["think"] = think
            m.setdefault("length_lambda", 0.7 if kind == "lcb_exec" else 0.0)
            rows.append({"id": f"{r['id']}#{'T' if think else 'N'}",
                         "source": r.get("source", kind),
                         "prompt": r["prompt"], "gold": str(r.get("gold") or ""),
                         "meta": m})
        cost += n * a.group * TOK[(kind, think)]

    rng.shuffle(rows)
    hours = cost / RATE
    n_lcb = sum(1 for r in rows if r["meta"]["reward_kind"] == "lcb_exec")
    n_think = sum(1 for r in rows if r["meta"]["think"])
    print(f"rows: {len(rows)}  (lcb {n_lcb} = {100*n_lcb/len(rows):.0f}%, "
          f"replay {len(rows)-n_lcb} = {100*(len(rows)-n_lcb)/len(rows):.0f}%)")
    print(f"thinking rows: {n_think} = {100*n_think/len(rows):.0f}%  |  "
          f"no-think: {len(rows)-n_think}")
    comp: dict[str, int] = {}
    for r in rows:
        comp[f"{r['meta']['reward_kind']}/{'think' if r['meta']['think'] else 'nothink'}"] = \
            comp.get(f"{r['meta']['reward_kind']}/{'think' if r['meta']['think'] else 'nothink'}", 0) + 1
    print("composition:", dict(sorted(comp.items())))
    print(f"PRICED at G={a.group}: {cost/1e6:.2f}M tokens -> ~{hours:.1f} h "
          f"at run2's realised {RATE//1000}k tok/h")
    if hours > a.hours_budget:
        sys.exit(f"REFUSE: {hours:.1f} h exceeds --hours-budget {a.hours_budget}. "
                 "An unaffordable pool must be rejected here, not discovered 20 h into "
                 "the run. Lower --lcb (it dominates: 48k tok/problem) or --mbpp-think.")
    pathlib.Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
