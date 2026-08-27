#!/usr/bin/env python3
"""Is the CURRENT lcb scorer the b606-fixed one, or the pre-fix one?

conciseness_report.py infers the scorer version from WHICH FIELD is present
(`passed_b606` -> fixed, `passed` -> original). That proxy is wrong for a FRESH run:
a run made after the fix writes plain `passed` using the fixed scorer, and gets
mislabelled as pre-fix -- which is what split the new OmniMerge-v4 row into its own
basis group.

The honest test is a control, not an inference: take an arm that HAS both fields,
re-score its stored completions with the scorer in the repo right now, and see which
of the two banked columns it reproduces.

  current == passed_b606  -> repo scorer IS the fixed one; a fresh run is comparable
  current == passed       -> repo scorer is PRE-fix; the new row is NOT comparable
  neither                 -> a third scorer version; everything needs re-basing
"""
import json
import pathlib
import sys

sys.path.insert(0, "/srv/ml/repos/omnimergekit")
from eval.lcb.lcb_helpers import clean_lcb_completion, load_lcb, score_lcb_problem  # noqa: E402

ROOT = pathlib.Path("/srv/ml/eval_results/qwen_suite/lcb_v6_77q")
CONTROL = sys.argv[1] if len(sys.argv) > 1 else "qwen184e_q6k"
TASKIDS = pathlib.Path("/srv/ml/repos/omnimergekit/eval/lcb/lcb_v6_77q_taskids.json")


def main() -> int:
    ids = json.load(TASKIDS.open())
    if isinstance(ids, dict):
        ids = ids.get("task_ids", list(ids))
    probs = {p["task_id"]: p for p in load_lcb(limit=0, task_ids=ids)}
    print(f"loaded {len(probs)} problems for the control")

    f = ROOT / CONTROL / "lcb_result.b606.samples.jsonl"
    if not f.exists():
        sys.exit(f"REFUSE: {f} missing -- control arm must carry BOTH columns")
    rows = [json.loads(ln) for ln in f.open()]
    if "passed_b606" not in rows[0] or "passed" not in rows[0]:
        sys.exit("REFUSE: control arm lacks both columns")

    agree_orig = agree_b606 = 0
    diff_examples = []
    for r in rows:
        p = probs.get(r["doc_id"])
        if p is None:
            sys.exit(f"REFUSE: {r['doc_id']} not in the loaded problem set")
        code = clean_lcb_completion(r["completion"], p["starter_code"])
        ok, why = score_lcb_problem(code, p["public_tests"], p["method_name"], timeout=10.0)
        agree_orig += int(ok == bool(r["passed"]))
        agree_b606 += int(ok == bool(r["passed_b606"]))
        if ok != bool(r["passed_b606"]) and len(diff_examples) < 5:
            diff_examples.append((r["doc_id"], ok, r["passed_b606"], why[:60]))

    n = len(rows)
    print(f"\ncontrol arm: {CONTROL}  (n={n})")
    print(f"  current scorer agrees with `passed`      (pre-fix) : {agree_orig}/{n}")
    print(f"  current scorer agrees with `passed_b606` (fixed)   : {agree_b606}/{n}")
    banked_delta = sum(bool(r["passed"]) != bool(r["passed_b606"]) for r in rows)
    print(f"  the two banked columns differ on         : {banked_delta}/{n} problems")
    if diff_examples:
        print("\n  disagreements vs b606:")
        for d in diff_examples:
            print(f"    {d[0]:22s} current={d[1]} b606={d[2]}  {d[3]}")

    if banked_delta == 0:
        print("\nINCONCLUSIVE: the two banked columns are identical on this arm -- "
              "it cannot discriminate. Re-run with an arm where they differ.")
        return 2
    if agree_b606 == n:
        print("\nSCORER_IS_B606: fresh runs are comparable to the b606 cells. "
              "Merge the OmniMerge-v4 row into the main basis group.")
        return 0
    if agree_orig == n:
        print("\nSCORER_IS_PREFIX: the repo scorer is the OLD one -- the v4 row is NOT "
              "comparable and must be re-scored before tabulating.")
        return 1
    print("\nSCORER_IS_THIRD_VERSION: matches neither banked column exactly. "
          "Every LCB cell needs re-basing before any cross-arm claim.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
