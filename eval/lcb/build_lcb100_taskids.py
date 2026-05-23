#!/usr/bin/env python3
"""Build a deterministic LCB-medium-100 task-id list that is a strict SUPERSET
of the canonical LCB-medium-55 set.

The canonical 55 = medium + functional + contest_date >= 2024-10-01 (the
contamination-controlled set lcb_medium_55*.yaml use). That pool is only ~55,
so to reach ~100 we relax `min_date` earlier — which only ADDS earlier
medium-functional problems; the post-2024-10 ones still qualify, so the union
necessarily contains all 55. We list the canonical 55 FIRST, then the earliest
N extras, and write the ids to a json the lcb_medium_100*.yaml templates load
via `selection.task_ids_file`.

Determinism: load_lcb walks the release shards test{,2..6}.jsonl in fixed order
and appends in encounter order, so the id list is stable across machines given
the same dataset snapshot.

Usage:
    python build_lcb100_taskids.py [--target 100] [--relaxed-min-date 2023-06-01] \\
        [--out eval/lcb/lcb_medium_100_taskids.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lcb_helpers import load_lcb  # noqa: E402

CANON_MIN_DATE = "2024-10-01"      # the canonical 55q cutoff
BIG = 100000                        # "no cap" sentinel for load_lcb limit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--relaxed-min-date", default="2023-06-01",
                    help="Earlier cutoff used to source the extra problems.")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent
                                         / "lcb_medium_100_taskids.json"))
    args = ap.parse_args()

    canon = [p["task_id"] for p in load_lcb(
        limit=BIG, difficulty="medium", min_date=CANON_MIN_DATE, testtype="functional")]
    canon_set = set(canon)
    print(f"[build] canonical (post {CANON_MIN_DATE}): {len(canon)} problems")

    relaxed = [p["task_id"] for p in load_lcb(
        limit=BIG, difficulty="medium", min_date=args.relaxed_min_date,
        testtype="functional")]
    extras = [t for t in relaxed if t not in canon_set]
    print(f"[build] relaxed (post {args.relaxed_min_date}): {len(relaxed)} total, "
          f"{len(extras)} extra candidates")

    need = max(0, args.target - len(canon))
    chosen_extras = extras[:need]
    final = canon + chosen_extras

    # Guarantee the superset property explicitly (defensive).
    assert canon_set.issubset(set(final)), "canonical 55 not fully contained!"
    if len(final) < args.target:
        print(f"[build] WARNING: only {len(final)} available (< target "
              f"{args.target}); relax --relaxed-min-date further for more.")

    Path(args.out).write_text(json.dumps(final, indent=2))
    print(f"[build] wrote {len(final)} task_ids → {args.out}")
    print(f"[build]   canonical={len(canon)}  extras={len(chosen_extras)}  "
          f"superset_ok={canon_set.issubset(set(final))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
