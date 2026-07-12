#!/usr/bin/env python3
"""Wait for both lcb_v6_55 runs (v7-coder + coderx-STD16) to finish, then print a
comparison table with the medium-vs-hard breakdown. Difficulty per task_id comes
from the loader (single source of truth); pass/fail from each run's lcb_result.json."""
import json
import os
import sys
import time

RES = "/srv/ml/eval_results_lcb_v6/lcb_v6_55"
TASKIDS = "/srv/ml/repos/omnimergekit/eval/lcb/lcb_v6_55_taskids.json"
RUNS = [("v7-coder", "v7coder-q6"), ("coderx-STD16(c4l3)", "cx16-c4l3")]

sys.path.insert(0, "/srv/ml/repos/omnimergekit/eval/lcb")
sys.path.insert(0, "/srv/ml/repos/omnimergekit/eval")

ids = json.load(open(TASKIDS))


def difficulty_map():
    from lcb_helpers import load_lcb
    probs = load_lcb(limit=999, task_ids=ids)
    return {p["task_id"]: p.get("difficulty", "?") for p in probs}


def per_task_pass(run_dir):
    """Best-effort: pull {task_id: passed} from lcb_result.json."""
    rj = os.path.join(RES, run_dir, "lcb_result.json")
    if not os.path.exists(rj):
        return {}
    try:
        d = json.load(open(rj))
    except Exception:
        return {}
    items = None
    for k in ("problems", "per_problem", "results", "details"):
        if isinstance(d.get(k), list):
            items = d[k]
            break
    if items is None and isinstance(d, list):
        items = d
    out = {}
    for it in items or []:
        if not isinstance(it, dict):
            continue
        tid = it.get("task_id") or it.get("id") or it.get("question_id")
        passed = it.get("passed")
        if passed is None:
            passed = it.get("pass")
        if passed is None and "pass@1" in it:
            passed = it.get("pass@1", 0) > 0
        if tid is not None:
            out[str(tid)] = bool(passed)
    return out


def wait_summary(run_dir, max_s=3600):
    f = os.path.join(RES, run_dir, "summary.json")
    t0 = time.time()
    while time.time() - t0 < max_s:
        if os.path.exists(f):
            return f
        time.sleep(30)
    return f if os.path.exists(f) else None


dmap = difficulty_map()
n_med = sum(1 for v in dmap.values() if v == "medium")
n_hard = sum(1 for v in dmap.values() if v == "hard")
print(f"[v6-55] task-id difficulty map: medium={n_med} hard={n_hard} total={len(dmap)}\n")

print("waiting for both runs to finish ...")
rows = []
for label, rd in RUNS:
    f = wait_summary(rd)
    score = None
    if f:
        try:
            score = json.load(open(f)).get("score")
        except Exception:
            pass
    ptp = per_task_pass(rd)
    mp = mt = hp = ht = 0
    for tid, diff in dmap.items():
        if tid not in ptp:
            continue
        ok = ptp[tid]
        if diff == "medium":
            mt += 1
            mp += int(ok)
        elif diff == "hard":
            ht += 1
            hp += int(ok)
    rows.append((label, score, mp, mt, hp, ht))

print("\n================ lcb_v6_55 — v7-coder vs coderx-STD16(code4/lcb3) ================")
print(f"{'model':<24}{'overall':>10}{'medium':>12}{'hard':>10}")
for label, score, mp, mt, hp, ht in rows:
    sc = f"{score*100:.2f}%" if isinstance(score, (int, float)) else "n/a"
    med = f"{mp}/{mt}" if mt else "?"
    hrd = f"{hp}/{ht}" if ht else "?"
    print(f"{label:<24}{sc:>10}{med:>12}{hrd:>10}")
print("\nanchors (saturated lcb_medium_55_v4): v7-coder ~93-96%  ·  coderx(c4l3) 92.73%")
print("###### V6_55_COMPARE_DONE ######")
