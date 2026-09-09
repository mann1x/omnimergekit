#!/usr/bin/env python3
"""Assemble grpo_pool_v4_budgeted.jsonl: a 60/40 driver/replay pool, 345 rows.

Knowledge doc (READ FIRST, keep updated): docs/METHOD_grpo_efficiency.md

WHY V4 EXISTS
-------------
v3 is 248 rows (128 manic + 96 gpqa + 24 lcb_easy). At `grad_accum 16` that is
62 steps/epoch, so a 150-step run is 2.4 epochs. v4 raises the pool to 345 so the
SAME 150-step run lands at 1.7 epochs -- AN's grpo_v8_v2 ran 2.0, and it is the only
brevity arm in this house with a significant length slope (OLS t=-2.43 / 446 steps).

So v4 is NOT "more data to train longer on". It is more data so a SHORT run sits at a
sane epoch count. The step budget is fixed by wall-clock; the pool size is what
converts that budget into epochs.

COMPOSITION -- driver supply is the binding constraint
------------------------------------------------------
  driver  manic-arm-contrast  128   (97 mined + 31 authored) -- EXHAUSTED, all of it
  driver  lcb_v6_easy          79   (v3 used 24; this takes all 79)
  replay  gpqa_main_minus_dia 138   (v3 used 96; 250 available)
                              ---
                              345    driver 207 (60.0%) / replay 138 (40.0%)

v4 is a strict SUPERSET of v3, so a v3 run and a v4 run remain comparable on the
shared rows. The gpqa top-up is seeded, so the same 42 rows are chosen every rebuild.

HOLDOUT -- re-derived here, never trusted from the source pool
--------------------------------------------------------------
The source pools were built by gated builders, but a gate that ran once upstream
proves nothing about THIS file. Both holdouts are re-derived from the original
datasets and intersected with the output, and the script REFUSES on any leak:

  * GPQA Diamond, two independent ways (Record ID and question hash). They must
    agree; a disagreement means the holdout is not well defined and is fatal.
  * Every *taskids.json under eval/lcb/ -- globbed, not enumerated, so a new frozen
    eval list is held out the day it lands.

[[feedback_withhold_by_tier_census_not_by_program_name]] -- a router-calib corpus
once carried 80 GPQA *Diamond* rows because it was filtered by program name.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import pathlib
import random
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
EFF = REPO / "eval" / "efficiency"

# Per-tier lambda/budget carried over from v3 so v4 is self-consistent. The runner
# overrides lambda globally (--length-lambda) and budgets (--budgets), but a row that
# reaches the reward with lambda>0 and no budget is a hard REFUSE, so they must be set.
TIER = {
    "manic-arm-contrast":      {"lam": 0.3, "budget": 1213},
    "gpqa_main_minus_diamond": {"lam": 0.1, "budget": 839},
    "lcb_v6_easy":             {"lam": 0.1, "budget": 2635},
}


def qhash(q: str) -> str:
    return hashlib.sha256(" ".join(q.split()).encode()).hexdigest()


# EXPLICIT roots only. A recursive glob from the filesystem anchor walks every
# mounted volume -- slow, crosses network mounts, and banned here. The first attempt
# at this script did exactly that and had to be killed.
# [[feedback_never_unscoped_find_across_volumes]]
HF_ROOTS = [
    "/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/huggingface/hub",
    str(pathlib.Path.home() / ".cache" / "huggingface" / "hub"),
    "/shared/hf/hub",
]


def resolve_gpqa() -> pathlib.Path:
    for root in HF_ROOTS:
        hits = glob.glob(f"{root}/datasets--Idavidrein--gpqa/snapshots/*/gpqa_main.csv")
        if hits:
            return pathlib.Path(sorted(hits)[0]).parent
    sys.exit("REFUSE: gpqa_main.csv not found under any of "
             + ", ".join(HF_ROOTS) + " -- cannot prove the Diamond holdout")


def diamond_holdout() -> set[str]:
    snap = resolve_gpqa()
    for n in ("gpqa_main.csv", "gpqa_diamond.csv"):
        if not (snap / n).exists():
            sys.exit(f"REFUSE: missing {snap / n} -- cannot prove the Diamond holdout")
    rd = lambda n: list(csv.DictReader((snap / n).open(encoding="utf-8")))  # noqa: E731
    main, dia = rd("gpqa_main.csv"), rd("gpqa_diamond.csv")
    d_ids = {r["Record ID"] for r in dia}
    d_hsh = {qhash(r["Question"]) for r in dia}
    # Two independent derivations of "main minus diamond". If they disagree, the
    # holdout is not well defined and no filter can be trusted.
    by_id = {r["Record ID"] for r in main if r["Record ID"] not in d_ids}
    by_h = {r["Record ID"] for r in main if qhash(r["Question"]) not in d_hsh}
    if by_id != by_h:
        sys.exit(f"REFUSE: Diamond-by-id and Diamond-by-hash disagree on "
                 f"{len(by_id ^ by_h)} rows -- holdout is not well defined")
    print(f"  GPQA: main={len(main)} diamond={len(dia)} usable={len(by_id)} "
          f"(id and hash derivations AGREE)")
    return d_ids


def lcb_holdout() -> set[str]:
    files = sorted(glob.glob(str(REPO / "eval" / "lcb" / "*taskids.json")))
    if not files:
        sys.exit("REFUSE: no *taskids.json under eval/lcb -- cannot prove holdout")
    out: set[str] = set()
    for f in files:
        ids = json.load(open(f))
        if isinstance(ids, dict):
            ids = ids.get("task_ids") or ids.get("ids") or []
        s = {str(x) for x in ids}
        print(f"  LCB frozen: {pathlib.Path(f).name:<34}{len(s):>5}")
        out |= s
    print(f"  LCB frozen: {'UNION':<34}{len(out):>5}")
    return out


def load(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def stamp(row: dict, src: str) -> dict:
    t = TIER[src]
    m = dict(row.get("meta") or {})
    m["length_lambda"] = t["lam"]
    m["length_budget"] = t["budget"]
    m.setdefault("length_budget_provenance",
                 "carried from grpo_pool_v3_budgeted.jsonl (P35 quantile of MEASURED "
                 "passing lengths, grpo_smoke_v3); runtime --budgets overrides")
    r = dict(row)
    r["meta"] = m
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EFF / "grpo_pool_v4_budgeted.jsonl"))
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--replay", type=int, default=138)
    a = ap.parse_args()

    print("=== holdout sets, re-derived from source ===")
    d_ids, lcb_hold = diamond_holdout(), lcb_holdout()

    v3 = load(EFF / "grpo_pool_v3_budgeted.jsonl")
    mixed = load(EFF / "gepo_mixed_with_efficiency.jsonl")
    by_src = collections.defaultdict(list)
    for d in mixed:
        by_src[d.get("source")].append(d)

    seen = {d["id"] for d in v3}
    out = [stamp(d, d["source"]) for d in v3]          # v4 is a SUPERSET of v3

    # lcb_v6_easy: take every row not already in v3
    add_lcb = [d for d in by_src["lcb_v6_easy"] if d["id"] not in seen]
    out += [stamp(d, "lcb_v6_easy") for d in add_lcb]

    # gpqa: seeded top-up to the requested replay count
    have = sum(1 for d in v3 if d["source"] == "gpqa_main_minus_diamond")
    pool = sorted((d for d in by_src["gpqa_main_minus_diamond"] if d["id"] not in seen),
                  key=lambda d: d["id"])
    random.Random(a.seed).shuffle(pool)
    need = max(0, a.replay - have)
    if need > len(pool):
        sys.exit(f"REFUSE: need {need} more gpqa rows, only {len(pool)} available")
    out += [stamp(d, "gpqa_main_minus_diamond") for d in pool[:need]]

    # ---- GATES ------------------------------------------------------------------
    ids = [d["id"] for d in out]
    if len(ids) != len(set(ids)):
        sys.exit(f"REFUSE: {len(ids) - len(set(ids))} duplicate ids")

    leak_g = sorted(d["id"] for d in out
                    if d["source"] == "gpqa_main_minus_diamond"
                    and d["id"].split("/")[-1].split("#")[0] in d_ids)
    leak_l = sorted(d["id"] for d in out
                    if str(d.get("source", "")).startswith("lcb_v6")
                    and d["id"].split("/")[-1].replace("#T", "") in lcb_hold)
    if leak_g or leak_l:
        print(f"\nREFUSE: {len(leak_g)} GPQA-Diamond and {len(leak_l)} frozen-LCB ids "
              f"leaked into the pool:")
        for x in (leak_g + leak_l)[:20]:
            print("   ", x)
        return 3

    c = collections.Counter(d["source"] for d in out)
    driver = c["manic-arm-contrast"] + c["lcb_v6_easy"]
    replay = c["gpqa_main_minus_diamond"]
    print(f"\n=== composition ({len(out)} rows) ===")
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {k:<28}{v:>5}")
    print(f"  {'DRIVER':<28}{driver:>5}  ({100 * driver / len(out):.1f}%)")
    print(f"  {'REPLAY':<28}{replay:>5}  ({100 * replay / len(out):.1f}%)")
    print(f"\nHOLDOUT_OK: 0/{len(d_ids)} Diamond and 0/{len(lcb_hold)} frozen-LCB ids "
          f"in a {len(out)}-row pool")

    with open(a.out, "w") as fh:
        for d in out:
            fh.write(json.dumps(d) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
