#!/usr/bin/env python3
"""
Rescore MBPP samples by stripping markdown fences before exec().

Sibling to scripts/rescore_humaneval_strip_fences.py. Same approach for
the bug-015 case: chat-trained models wrap code in ```python ... ```
which makes lm-eval's `exec()` scorer raise SyntaxError. Stripping
fences and re-running exec() yields the corrected pass_at_1.

MBPP differs from HumanEval in three ways:
  1. `filtered_resps[0]` is a flat str, not a nested list-of-list.
  2. The completion is the full function (no prompt prefix to drop).
  3. Scoring is `exec(test_setup_code + completion + "\\n".join(test_list))`
     — no `check(entry_point)` wrapper.

Usage:
    rescore_mbpp_strip_fences.py <samples.jsonl> [--timeout 10]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path


FENCE_OPEN = re.compile(r"^\s*```(?:python|py|python3)?\s*\n", re.IGNORECASE)
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_UNCLOSED = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)


def strip_fences(text: str) -> str:
    s = text
    s = THINK_BLOCK.sub("", s)
    s = THINK_OPEN_UNCLOSED.sub("", s)

    if "```" in s:
        first_fence = s.find("```")
        body_after_first = s[first_fence:]
        m_open = FENCE_OPEN.match(body_after_first)
        if m_open:
            tail = body_after_first[m_open.end():]
            close = tail.find("```")
            s = tail[:close] if close >= 0 else tail
        else:
            close = s.find("```")
            s = s[:close]
    if not s.endswith("\n"):
        s += "\n"
    return s


def _runner(program: str, q):
    try:
        exec(program, {})
        q.put(("pass", None))
    except BaseException as e:
        q.put(("fail", f"{type(e).__name__}: {e}"))


def run_program(program: str, timeout: float) -> tuple[bool, str]:
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_runner, args=(program, q))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        if p.is_alive():
            p.kill()
        return False, "timeout"
    try:
        verdict, msg = q.get_nowait()
        return verdict == "pass", (msg or "")
    except Exception:
        return False, "no-result"


def build_program(completion: str, doc: dict) -> str:
    setup = doc.get("test_setup_code", "") or ""
    test_list = doc.get("test_list", [])
    parts = []
    if setup.strip():
        parts.append(setup)
    parts.append(completion)
    parts.append("\n".join(test_list))
    return "\n".join(parts) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", type=Path)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.samples.exists():
        print(f"ERR samples missing: {args.samples}", file=sys.stderr)
        return 2

    rows = [json.loads(line) for line in args.samples.open()]
    if args.limit:
        rows = rows[: args.limit]
    total = len(rows)

    passed_old = passed_new = fenced = fixed = broken = 0
    fail_reasons: dict[str, int] = {}
    report = []

    for r in rows:
        doc = r["doc"]
        task_id = doc["task_id"]
        completion_raw = r["filtered_resps"][0]
        had_fence = "```" in completion_raw
        if had_fence:
            fenced += 1
        completion_clean = strip_fences(completion_raw)

        old_pass = bool(r.get("pass_at_1") in (1, 1.0, True))
        if old_pass:
            passed_old += 1

        program = build_program(completion_clean, doc)
        ok, reason = run_program(program, args.timeout)
        if ok:
            passed_new += 1
            if not old_pass:
                fixed += 1
        else:
            head = reason.split(":")[0] if reason else "fail"
            fail_reasons[head] = fail_reasons.get(head, 0) + 1
            if old_pass:
                broken += 1

        report.append({
            "task_id": task_id,
            "fenced": had_fence,
            "old_pass": old_pass,
            "new_pass": ok,
            "fail_reason": reason if not ok else None,
        })

        if not args.quiet:
            mark = "PASS" if ok else "FAIL"
            tag = "F" if had_fence else "."
            print(f"  [{mark}] {tag} {str(task_id):>12s} {'' if ok else reason[:80]}")

    pct_old = 100.0 * passed_old / total
    pct_new = 100.0 * passed_new / total
    print()
    print("=" * 64)
    print(f"Samples file: {args.samples.name}")
    print(f"Total tasks:  {total}")
    print(f"Fenced (stripped): {fenced}  Unfenced: {total - fenced}")
    print()
    print(f"  Original lm-eval pass_at_1:        {passed_old}/{total} = {pct_old:.2f}%")
    print(f"  Rescored pass_at_1 (fences stripped): {passed_new}/{total} = {pct_new:.2f}%")
    print(f"    delta: +{passed_new - passed_old} (fixed) / -{broken} (newly broken)")
    print()
    print("Failure reasons (rescored):")
    for k, v in sorted(fail_reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v}")

    if args.report:
        args.report.write_text(json.dumps({
            "samples_file": str(args.samples),
            "total": total, "fenced": fenced,
            "old_pass": passed_old, "new_pass": passed_new,
            "old_pct": pct_old, "new_pct": pct_new,
            "fail_reasons": fail_reasons,
            "per_task": report,
        }, indent=2))
        print(f"\n  Wrote report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
