#!/usr/bin/env python3
"""Build the frozen task-id list for the lcb_v6_55 template.

lcb_medium_55 is saturated: it is exactly the 55 medium+functional problems
after 2024-10, the newest/narrowest slice every strong code model has converged
on (v7-coder Q6_K ~96-98%). lcb_v6_55 is a *discriminating* replacement curated
from LiveCodeBench release_v6 (code_generation_lite, ~1055 problems):

  - scorer-compatible only: functional testtype + class-based starter
    (def X(self...)), matching lcb_helpers.score_lcb_problem's exec harness.
  - date window [2024-01-01, 2025-01-01): "not too new" (drops the 2025 frontier,
    contamination-status-uncertain / model-cutoff-sensitive) and "not too old"
    (drops 2023, highest train-set contamination risk).
  - 44 medium + 11 hard: "mostly not too hard", but the ~20% hard tail gives the
    headroom medium-only sets lack, so the bench separates strong models instead
    of re-saturating.
  - even month-spread via round-robin, deterministic (sorted by task_id within
    each month, months chronological). No randomness → fully reproducible.

Writes a JSON list of task_ids (chronological by contest_date) consumed by
templates/lcb_v6_55.yaml via selection.task_ids_file. The loader bypasses its
own difficulty/date filters when task_ids is set, so this list is the single
source of truth (and may mix difficulties).

Usage:
  python build_lcb_v6_55_taskids.py [--out eval/lcb/lcb_v6_55_taskids.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import re

from huggingface_hub import hf_hub_download

RELEASE_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl",
                 "test4.jsonl", "test5.jsonl", "test6.jsonl"]
SELF_RE = re.compile(r"def\s+(\w+)\s*\(\s*self")
WINDOW_LO = "2024-01-01"
WINDOW_HI = "2025-01-01"          # exclusive upper bound
N_MEDIUM = 44
N_HARD = 11


def compatible(row: dict) -> bool:
    """Functional testtype + class-based starter — the scorer's requirements."""
    raw = row.get("public_test_cases", "[]")
    if isinstance(raw, str):
        try:
            pub = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return False
    else:
        pub = raw or []
    if not pub or pub[0].get("testtype") != "functional":
        return False
    return bool(SELF_RE.search(row.get("starter_code", "") or ""))


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for fn in RELEASE_FILES:
        try:
            path = hf_hub_download(repo_id="livecodebench/code_generation_lite",
                                   repo_type="dataset", filename=fn)
        except Exception as exc:                       # noqa: BLE001
            print(f"[build] skip {fn}: {exc}")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def task_id(row: dict) -> str:
    return f"lcb/{row.get('platform', '?')}/{row.get('question_id', '?')}"


def month(row: dict) -> str:
    return (row.get("contest_date") or "")[:7]


def round_robin_select(pool: list[dict], quota: int) -> list[dict]:
    """Even month-spread: queue each month's problems (sorted by task_id),
    then deal one per month in chronological order until the quota is met."""
    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for row in pool:
        buckets[month(row)].append(row)
    for key in buckets:
        buckets[key].sort(key=task_id)
    months = sorted(buckets)
    queues = {m: collections.deque(buckets[m]) for m in months}
    picked: list[dict] = []
    while len(picked) < quota and any(queues[m] for m in months):
        for m in months:
            if len(picked) >= quota:
                break
            if queues[m]:
                picked.append(queues[m].popleft())
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="eval/lcb/lcb_v6_55_taskids.json")
    args = ap.parse_args()

    rows = load_rows()
    comp = [r for r in rows if compatible(r)]
    in_window = [r for r in comp if WINDOW_LO <= (r.get("contest_date") or "")[:10] < WINDOW_HI]
    medium = [r for r in in_window if r.get("difficulty") == "medium"]
    hard = [r for r in in_window if r.get("difficulty") == "hard"]
    print(f"[build] release rows={len(rows)} compatible={len(comp)} "
          f"in-window[{WINDOW_LO},{WINDOW_HI})={len(in_window)} "
          f"(medium={len(medium)} hard={len(hard)})")

    sel_medium = round_robin_select(medium, N_MEDIUM)
    sel_hard = round_robin_select(hard, N_HARD)
    selected = sel_medium + sel_hard
    if len(selected) != N_MEDIUM + N_HARD:
        raise SystemExit(f"[build] FATAL got {len(selected)} (want {N_MEDIUM + N_HARD}); "
                         f"pool too small (medium={len(medium)} hard={len(hard)})")

    selected.sort(key=lambda r: ((r.get("contest_date") or "")[:10], task_id(r)))
    ids = [task_id(r) for r in selected]
    if len(set(ids)) != len(ids):
        raise SystemExit("[build] FATAL duplicate task_ids")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)
        f.write("\n")

    by_diff = collections.Counter(r.get("difficulty") for r in selected)
    by_month = collections.Counter(month(r) for r in selected)
    print(f"[build] wrote {len(ids)} task_ids -> {args.out}")
    print(f"[build] difficulty: {dict(by_diff)}")
    print("[build] month spread:")
    for m in sorted(by_month):
        print(f"  {m}: {by_month[m]}")


if __name__ == "__main__":
    main()
