#!/usr/bin/env python3
"""Offline LCB rescore under the bug-606 fix. No GPU, no server, no regeneration.

bug-606: clean_lcb_completion needed >= 2 fence-bearing lines to slice out the
code block (official LCB semantics: the slice between the LAST TWO fences). A
cap-truncated runaway has exactly ONE -- the unterminated opener -- and the old
fallback only stripped a TRAILING fence, so the opening ```python survived into
the scored code and made line 1 a SyntaxError by construction. A zero our own
extractor manufactured, banked as a model failure.

This is a RESCORE, not a re-run: LCB persisted the raw `completion` next to
`cleaned`, so the fixed extractor can be applied to the stored raw and the code
re-executed. (MPE had no such raw -- that is bug-607 -- which is why bug-604
forced a regeneration.)

RESULTS ARE SACRED. Nothing here overwrites an existing artifact. Each cell
gains, alongside its originals:
    lcb_result.b606.json           new score + full delta accounting
    lcb_result.b606.samples.jsonl  per-sample old/new cleaned + verdict

CONTROL (this is what makes the numbers trustworthy). Every sample is
re-executed, including the ones whose `cleaned` did not change a single byte.
Those MUST reproduce their banked verdict. If they do not, the rescore harness
itself is wrong -- different dataset draw, different scorer, a flaky timeout --
and the whole cell is reported as SUSPECT rather than banked. A rescore that
only re-runs the samples it expects to change cannot tell "the fix worked" from
"my harness disagrees with the original run".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/shared/dev/omnimergekit")
sys.path.insert(0, "/shared/dev/omnimergekit/eval")

from lcb.lcb_helpers import (  # noqa: E402
    clean_lcb_completion,
    load_lcb,
    score_lcb_problem,
)

_PROBLEM_CACHE: dict[str, dict] = {}


def load_problems(task_ids: list[str]) -> dict[str, dict]:
    """task_id -> problem dict (tests + method_name). Cached across cells:
    every cell in a bench dir is the SAME 77-problem draw, so one load serves
    all of them and the draw is identical by construction."""
    missing = [t for t in task_ids if t not in _PROBLEM_CACHE]
    if missing:
        for p in load_lcb(limit=0, task_ids=sorted(set(task_ids))):
            _PROBLEM_CACHE[p["task_id"]] = p
    return {t: _PROBLEM_CACHE[t] for t in task_ids if t in _PROBLEM_CACHE}


def rescore_cell(cell: Path, timeout: float, verbose: bool) -> dict | None:
    sf = cell / "lcb_result.samples.jsonl"
    if not sf.exists():
        return None
    rows = []
    for line in sf.open(errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return None

    probs = load_problems([r.get("task_id") or r.get("doc_id") or "" for r in rows])

    out_rows = []
    n = n_unresolved = 0
    changed = unchanged = 0
    ctrl_ok = ctrl_bad = 0          # control: unchanged `cleaned`, verdict match?
    fixed = broke = 0               # changed `cleaned`: fail->pass, pass->fail
    old_pass = new_pass = 0
    ctrl_bad_ids: list[str] = []
    fixed_ids: list[str] = []
    broke_ids: list[str] = []

    for r in rows:
        tid = r.get("task_id") or r.get("doc_id") or ""
        raw = r.get("completion") or ""
        old_clean = r.get("cleaned") or ""
        old_passed = bool(r.get("passed"))
        n += 1
        old_pass += int(old_passed)

        p = probs.get(tid)
        if p is None:
            # Cannot re-execute without the problem's tests. REFUSE to guess:
            # carry the banked verdict and count it, so an unresolved task can
            # never be silently scored as a failure.
            n_unresolved += 1
            new_pass += int(old_passed)
            out_rows.append({**r, "b606_status": "unresolved_task_id"})
            continue

        new_clean = clean_lcb_completion(raw, p["starter_code"])
        same = (new_clean == old_clean)
        ok, reason = score_lcb_problem(new_clean, p["public_tests"],
                                       p["method_name"], timeout=timeout)
        new_pass += int(ok)

        if same:
            unchanged += 1
            if ok == old_passed:
                ctrl_ok += 1
            else:
                ctrl_bad += 1
                if len(ctrl_bad_ids) < 10:
                    ctrl_bad_ids.append(tid)
        else:
            changed += 1
            if ok and not old_passed:
                fixed += 1
                if len(fixed_ids) < 20:
                    fixed_ids.append(tid)
            elif old_passed and not ok:
                broke += 1
                if len(broke_ids) < 20:
                    broke_ids.append(tid)

        out_rows.append({**r, "cleaned_b606": new_clean, "passed_b606": ok,
                         "reason_b606": reason,
                         "b606_status": "unchanged" if same else "recleaned"})
        if verbose and not same and ok != old_passed:
            print(f"      {tid}: {old_passed} -> {ok}")

    # A control failure means the harness disagrees with the original run on
    # input it did not touch. The cell's new score is then not a measurement.
    suspect = ctrl_bad > 0

    res = {
        "cell": cell.name,
        "n": n,
        "pass_at_1_old": round(old_pass / n, 4) if n else None,
        "pass_at_1_b606": round(new_pass / n, 4) if n else None,
        "delta_pp": round(100.0 * (new_pass - old_pass) / n, 2) if n else None,
        "n_pass_old": old_pass,
        "n_pass_b606": new_pass,
        "recleaned": changed,
        "unchanged": unchanged,
        "control_reproduced": ctrl_ok,
        "control_mismatch": ctrl_bad,
        "control_mismatch_ids": ctrl_bad_ids,
        "fixed_fail_to_pass": fixed,
        "regressed_pass_to_fail": broke,
        "fixed_ids": fixed_ids,
        "regressed_ids": broke_ids,
        "unresolved_task_ids": n_unresolved,
        "verdict": "SUSPECT_control_mismatch" if suspect else "OK",
        "bug": "bug-606",
        "extractor": "clean_lcb_completion@single-fence-opener-fix",
    }
    (cell / "lcb_result.b606.json").write_text(json.dumps(res, indent=2))
    with (cell / "lcb_result.b606.samples.jsonl").open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benches", nargs="+", help="bench dirs holding per-cell subdirs")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    all_res = []
    for b in a.benches:
        bp = Path(b)
        print(f"\n### {bp}", flush=True)
        for cell in sorted(p for p in bp.iterdir() if p.is_dir()):
            print(f"  - {cell.name} ...", end="", flush=True)
            r = rescore_cell(cell, a.timeout, a.verbose)
            if r is None:
                print(" no samples, skip")
                continue
            print(f" {r['pass_at_1_old']:.4f} -> {r['pass_at_1_b606']:.4f} "
                  f"({r['delta_pp']:+.2f}pp)  recleaned={r['recleaned']} "
                  f"fixed={r['fixed_fail_to_pass']} regressed={r['regressed_pass_to_fail']} "
                  f"ctrl={r['control_reproduced']}/{r['control_reproduced'] + r['control_mismatch']}"
                  f"{'  ** ' + r['verdict'] if r['verdict'] != 'OK' else ''}", flush=True)
            all_res.append({"bench": bp.name, **r})

    print("\n" + "=" * 108)
    print(f"{'bench':<18} {'cell':<28} {'n':>4} {'old':>7} {'b606':>7} {'Δpp':>7} "
          f"{'recln':>6} {'fix':>4} {'reg':>4} {'ctrl':>9} verdict")
    print("-" * 108)
    for r in all_res:
        ctrl = f"{r['control_reproduced']}/{r['control_reproduced'] + r['control_mismatch']}"
        print(f"{r['bench']:<18} {r['cell']:<28} {r['n']:>4} "
              f"{r['pass_at_1_old']:>7.4f} {r['pass_at_1_b606']:>7.4f} {r['delta_pp']:>+7.2f} "
              f"{r['recleaned']:>6} {r['fixed_fail_to_pass']:>4} "
              f"{r['regressed_pass_to_fail']:>4} {ctrl:>9} {r['verdict']}")
    print("-" * 108)
    bad = [r for r in all_res if r["verdict"] != "OK"]
    print(f"cells: {len(all_res)}   SUSPECT: {len(bad)}"
          + (f"  -> {[r['cell'] for r in bad]}" if bad else ""))
    print("ctrl = samples whose `cleaned` is byte-identical and whose banked verdict")
    print("       reproduced. Any mismatch means the harness disagrees with the")
    print("       original run on untouched input -> the cell is not a measurement.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
