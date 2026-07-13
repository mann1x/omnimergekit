#!/usr/bin/env python3
"""Build the frozen task-id list for the LCB-targeted calibration corpus (Qwen coder variant).

This is the DISJOINT complement of the lcb_v6_55 eval set: every scorer-compatible
(functional + class-based starter) LiveCodeBench release_v6 problem in the same
[2024-01-01, 2025-01-01) window and difficulty band (medium + hard) that is NOT one
of the 55 lcb_v6_55 evaluation task-ids. Using a disjoint set is mandatory — calibrating
the targeted prune on the exact problems it will later be scored on would overfit the
drop map to the test set.

The teacher (256e) generates PASSING full-CoT solutions on these problems; those PASS
trajectories become the `targeted_lcb` competence channel (mirrors the Gemma v7-coder
128e-PASS `targeted_lcb_medium_55` channel — see docs/T17_v5_targeted_pruning_strategy.md).

Reuses build_lcb_v6_55_taskids.py's exact scorer-compat filter so every calib problem
is guaranteed gradeable (PASS/FAIL harvest depends on it).

Usage:
  python build_lcb_calib_taskids.py [--out eval/lcb/lcb_calib_taskids.json] \
      [--eval-ids eval/lcb/lcb_v6_55_taskids.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lcb_v6_55_taskids import (  # noqa: E402
    WINDOW_HI, WINDOW_LO, compatible, load_rows, task_id,
)


def month(row: dict) -> str:
    return (row.get("contest_date") or "")[:7]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="eval/lcb/lcb_calib_taskids.json")
    ap.add_argument("--eval-ids", default="eval/lcb/lcb_v6_55_taskids.json",
                    help="Eval task-ids to EXCLUDE (the calib set is their disjoint complement).")
    args = ap.parse_args()

    with open(args.eval_ids) as f:
        eval_ids = set(json.load(f))
    print(f"[calib] excluding {len(eval_ids)} lcb_v6_55 eval task-ids")

    rows = load_rows()
    comp = [r for r in rows if compatible(r)]
    in_window = [r for r in comp
                 if WINDOW_LO <= (r.get("contest_date") or "")[:10] < WINDOW_HI
                 and r.get("difficulty") in ("medium", "hard")]
    disjoint = [r for r in in_window if task_id(r) not in eval_ids]
    disjoint.sort(key=lambda r: ((r.get("contest_date") or "")[:10], task_id(r)))
    ids = [task_id(r) for r in disjoint]
    if len(set(ids)) != len(ids):
        raise SystemExit("[calib] FATAL duplicate task_ids")
    # every excluded eval id must actually be gone
    leaked = eval_ids & set(ids)
    if leaked:
        raise SystemExit(f"[calib] FATAL {len(leaked)} eval ids leaked into calib set: {sorted(leaked)[:5]}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)
        f.write("\n")

    by_diff = collections.Counter(r.get("difficulty") for r in disjoint)
    by_month = collections.Counter(month(r) for r in disjoint)
    print(f"[calib] release rows={len(rows)} compatible={len(comp)} "
          f"in-window medium+hard={len(in_window)} disjoint-from-eval={len(ids)}")
    print(f"[calib] wrote {len(ids)} task_ids -> {args.out}")
    print(f"[calib] difficulty: {dict(by_diff)}")
    print("[calib] month spread: " + ", ".join(f"{m}={by_month[m]}" for m in sorted(by_month)))


if __name__ == "__main__":
    main()
