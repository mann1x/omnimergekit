#!/usr/bin/env python3
"""
Rescore HumanEval samples by stripping markdown fences before exec().

Bug-015: chat-trained / reasoning models wrap code in ```python ... ```
fences when prompted via /v1/completions. lm-eval's code_eval scorer
runs `exec(prompt + completion + test + check())` and the trailing
``` causes SyntaxError, scoring the answer as failed even when the code
inside is correct.

This script reads the lm-eval samples_humaneval JSONL, strips markdown
fences from each completion, and re-runs exec() locally with a per-test
timeout. Produces a corrected pass@1 number directly comparable to the
unfenced runs (Qwen sources where the same args produced clean code).

Usage:
    rescore_humaneval_strip_fences.py <samples.jsonl> [--timeout 10]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path


FENCE_OPEN = re.compile(r"^\s*```(?:python|py|python3)?\s*\n", re.IGNORECASE)
FENCE_CLOSE = re.compile(r"\n```\s*$", re.IGNORECASE)
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_UNCLOSED = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)

GEMMA_CHANNEL_THOUGHT = re.compile(
    r"<\|channel>(?:thought|analysis|thinking).*?(?=<\|channel>|$)",
    re.DOTALL | re.IGNORECASE,
)
GEMMA_CHANNEL_FINAL_HEADER = re.compile(
    r"<\|channel>(?:final|answer)\s*<channel\|>",
    re.IGNORECASE,
)
GEMMA_LEFTOVER_MARKERS = re.compile(r"<\|channel>|<channel\|>", re.IGNORECASE)


def strip_fences(text: str) -> str:
    """Strip reasoning markers and ```python fences, return only the code body.

    Pipeline (in order):
      1. Remove <think>...</think> blocks (closed + trailing-unclosed) — Qwen-style.
      2. Remove Gemma 4 channel-token blocks: <|channel>thought<channel|>...
         until next <|channel>; strip <|channel>final<channel|> header so the
         final section content is kept; strip leftover <|channel> / <channel|>.
      3. Fence handling, two cases:
         a. Text STARTS with ```lang? — treat as opening fence: drop the open
            line, keep content up to the next ``` (closing).
         b. Text has body code first, then ``` — treat as closing fence: keep
            preamble (the code), strip from ``` onward.
    """
    s = text

    s = THINK_BLOCK.sub("", s)
    s = THINK_OPEN_UNCLOSED.sub("", s)

    s = GEMMA_CHANNEL_THOUGHT.sub("", s)
    s = GEMMA_CHANNEL_FINAL_HEADER.sub("", s)
    s = GEMMA_LEFTOVER_MARKERS.sub("", s)

    if "```" in s:
        m_open_at_start = FENCE_OPEN.match(s)
        if m_open_at_start:
            tail = s[m_open_at_start.end():]
            close = tail.find("```")
            s = tail[:close] if close >= 0 else tail
        else:
            first_fence = s.find("```")
            s = s[:first_fence]

    if not s.endswith("\n"):
        s += "\n"
    return s


def _runner(program: str, q):
    try:
        glb: dict = {}
        exec(program, glb)
        q.put(("pass", None))
    except BaseException as e:
        q.put(("fail", f"{type(e).__name__}: {e}"))


def run_program(program: str, timeout: float) -> tuple[bool, str]:
    """Run `program` in a subprocess with timeout. Returns (passed, reason)."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_runner, args=(program, q))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate()
        p.join(timeout=1.0)
        if p.is_alive():
            p.kill()
        return False, "timeout"
    try:
        verdict, msg = q.get_nowait()
        return verdict == "pass", (msg or "")
    except Exception:
        return False, "no-result"


def build_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    return f"{prompt}{completion}\n\n{test}\n\ncheck({entry_point})\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", type=Path)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report", type=Path, default=None,
                    help="Write per-task JSON report (task_id, passed, fenced, reason).")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--from-raw", action="store_true",
        help="Use resps[0][0] (raw model output) instead of filtered_resps[0][0]. "
             "Required for humaneval_instruct samples where the task filter has "
             "already partially-processed the response.",
    )
    ap.add_argument(
        "--pass-key", default="pass@1",
        help="Sample-level pass field (humaneval='pass@1', humaneval_instruct='pass@1').",
    )
    args = ap.parse_args()

    if not args.samples.exists():
        print(f"ERR samples file missing: {args.samples}", file=sys.stderr)
        return 2

    rows = []
    with args.samples.open() as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    total = len(rows)
    passed_old = 0
    passed_new = 0
    fenced = 0
    fixed = 0
    broken = 0
    fail_reasons: dict[str, int] = {}
    report = []

    for r in rows:
        doc = r["doc"]
        prompt = doc["prompt"]
        test = doc["test"]
        entry_point = doc["entry_point"]
        task_id = doc["task_id"]

        if args.from_raw:
            resps0 = r["resps"][0]
            raw = resps0[0] if isinstance(resps0, list) else resps0
        else:
            raw = r["filtered_resps"][0][0]
        if raw.startswith(prompt):
            completion_raw = raw[len(prompt):]
        else:
            completion_raw = raw

        had_fence = "```" in completion_raw
        if had_fence:
            fenced += 1
        completion_clean = strip_fences(completion_raw)

        old_pass = bool(r.get(args.pass_key) in (1, 1.0, True))
        if old_pass:
            passed_old += 1

        program = build_program(prompt, completion_clean, test, entry_point)
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
            print(f"  [{mark}] {tag} {task_id:24s} {'' if ok else reason[:80]}")

    pct_old = 100.0 * passed_old / total
    pct_new = 100.0 * passed_new / total

    print()
    print("=" * 64)
    print(f"Samples file: {args.samples.name}")
    print(f"Total tasks:  {total}")
    print(f"Fenced (stripped): {fenced}  Unfenced: {total - fenced}")
    print()
    print(f"  Original pass@1 (lm-eval, fenced exec'd):  {passed_old}/{total} = {pct_old:.2f}%")
    print(f"  Rescored pass@1 (fences stripped):         {passed_new}/{total} = {pct_new:.2f}%")
    print(f"    delta: +{passed_new - passed_old} (fixed) / -{broken} (newly broken)")
    print()
    print("Failure reasons (rescored):")
    for k, v in sorted(fail_reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v}")

    if args.report:
        args.report.write_text(json.dumps({
            "samples_file": str(args.samples),
            "total": total,
            "fenced": fenced,
            "old_pass": passed_old,
            "new_pass": passed_new,
            "old_pct": pct_old,
            "new_pct": pct_new,
            "fail_reasons": fail_reasons,
            "per_task": report,
        }, indent=2))
        print(f"\n  Wrote report: {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
