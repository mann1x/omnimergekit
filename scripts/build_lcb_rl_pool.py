#!/usr/bin/env python3
"""Build the LiveCodeBench RL prompt pool for correctness-gated brevity training.

WHY THIS EXISTS
---------------
CoderX (Qwen3.6-27B-A3B-CoderX) outscores its A3B-Coder sibling on LCB-v6-77q
(0.7273 vs 0.6104) but is 7-10x more verbose, and that verbosity is what makes it
fail in the agentic harness. The fix is an RL pass with a CORRECTNESS-GATED LENGTH
reward: among rollouts that still pass the tests, reward the shorter one.

    reward = (1 - lambda * min(ntok / budget, 1))  if tests pass  else 0

That is the same shape the an-finetune GEPO run used (simpo/train_grpo_e2b.py), but
that pool was gsm8k/math/aime with a NUMERIC gold and `grade_numeric` as the verifier.
There is no LCB GEPO pool -- code needs an execution verifier, which is what
scripts/lcb_brevity_reward.py adds. This script builds the prompt side.

CONTAMINATION -- the reason this is not just "take the 77"
----------------------------------------------------------
The 12 problems where Coder passes and CoderX fails ARE the eval. Training on them
turns lcb_v6_77q into a train-on-test number and voids it as a decision gate, along
with every other frozen LCB list that shares problems. So the pool is built from
release_v6 MINUS the union of every frozen eval task-id list under eval/lcb/, and
the script REFUSES (exit 3) if a single eval id survives into the pool.

The brevity signal does not need those 12 anyway. On the 35 problems where BOTH
models pass, CoderX still burns 9.78x the tokens (p50 11,882 vs 1,433) -- the
verbosity is present on its WINS, so it is trainable on held-out problems without
trading away the correctness CoderX actually gained.

OUTPUT SCHEMA -- a superset of the an-finetune GEPO pool ({id, source, prompt, gold,
meta}), so the existing launcher reads it unchanged. `gold` is unused by the code
reward (there is no single right string); the verifier reads meta.tests /
meta.method_name / meta.starter_code instead.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.lcb.lcb_helpers import LCB_INSTRUCT_TEMPLATE, load_lcb  # noqa: E402

FROZEN_DIR = REPO / "eval" / "lcb"


def frozen_eval_ids() -> tuple[set[str], dict[str, int]]:
    """Every task-id that appears in ANY frozen eval list. All of them are holdout.

    Deliberately globbed rather than enumerated: a new *_taskids.json is a new eval
    set, and it must be excluded the day it lands, not the day someone remembers to
    edit this list.
    """
    per_file: dict[str, int] = {}
    ids: set[str] = set()
    files = sorted(FROZEN_DIR.glob("*taskids.json"))
    if not files:
        sys.exit(f"REFUSE: no *taskids.json under {FROZEN_DIR} -- cannot prove holdout")
    for f in files:
        raw = json.load(f.open())
        got = raw.get("task_ids", list(raw)) if isinstance(raw, dict) else raw
        per_file[f.name] = len(got)
        ids |= set(got)
    return ids, per_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "eval" / "lcb" / "lcb_rl_pool.jsonl"))
    ap.add_argument("--difficulties", default="easy,medium,hard")
    ap.add_argument("--min-date", default="2024-01-01",
                    help="matches the lcb_v6_77q build window")
    ap.add_argument("--limit-per-difficulty", type=int, default=100000)
    ap.add_argument("--min-tests", type=int, default=1,
                    help="drop problems with fewer public tests than this -- the reward "
                         "is only as trustworthy as the tests behind it")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    holdout, per_file = frozen_eval_ids()
    print("=== frozen eval task-id lists (ALL held out) ===")
    for name, n in per_file.items():
        print(f"  {name:34s} {n:5d}")
    print(f"  {'union':34s} {len(holdout):5d}\n")

    rows: list[dict] = []
    seen: set[str] = set()
    for diff in args.difficulties.split(","):
        diff = diff.strip()
        probs = load_lcb(limit=args.limit_per_difficulty, difficulty=diff,
                         min_date=args.min_date, testtype="functional")
        kept = 0
        for p in probs:
            tid = p["task_id"]
            if tid in holdout or tid in seen:
                continue
            if len(p["public_tests"]) < args.min_tests:
                continue
            seen.add(tid)
            kept += 1
            rows.append({
                "id": tid,
                "source": f"lcb_v6_{diff}",
                "prompt": LCB_INSTRUCT_TEMPLATE.format(
                    question=p["question_content"], starter=p["starter_code"]),
                "gold": "",                       # unused: the verifier executes tests
                "meta": {
                    "difficulty": p["difficulty"],
                    "contest_date": p["contest_date"],
                    "starter_code": p["starter_code"],
                    "method_name": p["method_name"],
                    "tests": p["public_tests"],
                    "n_tests": len(p["public_tests"]),
                },
            })
        print(f"  {diff:8s} -> kept {kept}")

    # --- the gate: a single surviving eval id makes every downstream LCB cell a lie
    leaked = sorted({r["id"] for r in rows} & holdout)
    if leaked:
        print(f"\nREFUSE: {len(leaked)} frozen eval task-ids leaked into the pool:")
        for t in leaked[:20]:
            print(f"  {t}")
        return 3
    if not rows:
        print("\nREFUSE: empty pool")
        return 4

    random.Random(args.seed).shuffle(rows)
    out = pathlib.Path(args.out)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    by_diff: dict[str, int] = {}
    tests = []
    for r in rows:
        by_diff[r["source"]] = by_diff.get(r["source"], 0) + 1
        tests.append(r["meta"]["n_tests"])
    tests.sort()
    print(f"\nHOLDOUT_OK: 0/{len(holdout)} frozen eval ids in a {len(rows)}-prompt pool")
    print(f"composition: {by_diff}")
    print(f"public tests per problem: min {tests[0]}, p50 {tests[len(tests) // 2]}, "
          f"max {tests[-1]}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
