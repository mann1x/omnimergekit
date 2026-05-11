# Auto-extracted LCB helpers from Mythic-RDT/humaneval_smoke.py
from __future__ import annotations
import json, re, sys, time, signal, multiprocessing as mp
from dataclasses import dataclass, asdict
from pathlib import Path
from huggingface_hub import hf_hub_download

FENCED_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

TRAILING_FENCE_RE = re.compile(r"\n?```\s*$", re.IGNORECASE)

LCB_INSTRUCT_TEMPLATE = (
    "Solve the following Python coding problem. Respond with ONLY the "
    "completed Solution class in a Python markdown block, no explanations.\n\n"
    "{question}\n\n"
    "```python\n{starter}\n```"
)

def load_lcb(limit: int, difficulty: str = "medium",
             min_date: str = "2024-10-01",
             testtype: str = "functional") -> list[dict]:
    """Load LiveCodeBench problems filtered to function_call style.

    Returns a list of normalized problem dicts:
      - task_id, question_content, starter_code, method_name, difficulty,
        public_tests (list of {input, output, testtype}).

    Filtering:
      - difficulty match (default "medium" — easy is too easy, hard runs are slow)
      - contest_date >= min_date (contamination control: defaults post-2024-10
        which is after DS-Coder-V2-Lite's training cutoff)
      - testtype == "functional" (skip stdin/stdout style for smoke; full LCB
        eval at v4 end will use lcb-runner which handles stdin properly)

    Note on loading: datasets>=4.0 dropped support for trust_remote_code-based
    dataset scripts and LCB ships its data behind such a script. We bypass it
    by downloading the underlying JSONL release files directly via
    huggingface_hub. As of 2026-04 the release set is test{,2..6}.jsonl and
    contains ~1055 problems total; the smoke filter typically yields ~55
    medium / ~34 easy / ~38 hard candidates post-2024-10.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[lcb] WARN: `huggingface_hub` not installed; skipping LCB.")
        return []
    print(f"[lcb] loading livecodebench/code_generation_lite "
          f"(difficulty={difficulty}, min_date={min_date}, testtype={testtype})...")
    release_files = ["test.jsonl", "test2.jsonl", "test3.jsonl",
                     "test4.jsonl", "test5.jsonl", "test6.jsonl"]
    out: list[dict] = []
    for fn in release_files:
        try:
            path = hf_hub_download(
                repo_id="livecodebench/code_generation_lite",
                repo_type="dataset",
                filename=fn,
            )
        except Exception as exc:
            print(f"[lcb]   skip {fn}: {exc}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("difficulty") != difficulty:
                    continue
                contest_date = (row.get("contest_date") or "")[:10]
                if contest_date and contest_date < min_date:
                    continue
                public_raw = row.get("public_test_cases", "[]")
                if isinstance(public_raw, str):
                    try:
                        public = json.loads(public_raw)
                    except (json.JSONDecodeError, ValueError):
                        continue
                else:
                    public = public_raw or []
                if not public:
                    continue
                if public[0].get("testtype") != testtype:
                    continue
                starter = row.get("starter_code", "") or ""
                m = re.search(r"def\s+(\w+)\s*\(\s*self", starter)
                if not m:
                    continue
                out.append({
                    "task_id": f"lcb/{row.get('platform','?')}/{row.get('question_id','?')}",
                    "question_content": row.get("question_content", ""),
                    "starter_code": starter,
                    "method_name": m.group(1),
                    "difficulty": row.get("difficulty", ""),
                    "contest_date": contest_date,
                    "public_tests": public,
                })
                if len(out) >= limit:
                    break
        if len(out) >= limit:
            break
    print(f"[lcb] loaded {len(out)} problems "
          f"(difficulty={difficulty}, post {min_date}, functional)")
    return out

def clean_lcb_completion(completion: str, starter_code: str) -> str:
    """Extract the Solution class from a fenced block; fall back to raw."""
    m = FENCED_BLOCK_RE.search(completion)
    if m:
        return m.group(1)
    # No fence found; strip trailing fence if partial, return as-is.
    return TRAILING_FENCE_RE.sub("", completion)

def _score_lcb_worker(code: str, tests: list, method_name: str, q):
    """Run inside child process. Imports are inside the function so they
    survive the fork in subprocesses without re-importing module globals."""
    import ast as _ast
    try:
        # Pre-populate namespace with common typing + stdlib symbols so
        # starter_code with `List[int]`, `Optional[str]`, etc. exec()s without
        # NameError (LCB starter signatures use these heavily).
        preamble = (
            "from typing import List, Dict, Tuple, Set, Optional, Union, "
            "Any, Callable, Iterator, Iterable, Sequence\n"
            "from collections import defaultdict, deque, Counter, OrderedDict\n"
            "from math import inf, gcd, floor, ceil, sqrt, log, log2, factorial\n"
            "from heapq import heappush, heappop, heapify, nlargest, nsmallest\n"
            "from bisect import bisect_left, bisect_right, insort\n"
            "from itertools import accumulate, combinations, permutations, product\n"
            "from functools import lru_cache, cache, reduce\n"
        )
        ns: dict = {}
        exec(preamble + code, ns)
        Solution = ns.get("Solution")
        if Solution is None:
            q.put(("fail", "no Solution class defined"))
            return
        for i, t in enumerate(tests):
            inp_str = (t.get("input") or "").strip()
            exp_str = (t.get("output") or "").strip()
            # LCB encodes input as Python literals. Two formats observed:
            #   - Single-line: the WHOLE string is one arg literal (e.g.
            #     "[1,2,3]" means one List[int] arg, NOT three int args).
            #   - Multi-line: each line is one positional arg literal.
            if "\n" in inp_str:
                args = []
                for line in inp_str.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        args.append(_ast.literal_eval(line))
                    except (ValueError, SyntaxError):
                        args.append(line)
                args = tuple(args)
            else:
                try:
                    args = (_ast.literal_eval(inp_str),)
                except (ValueError, SyntaxError):
                    args = (inp_str,)
            try:
                expected = _ast.literal_eval(exp_str)
            except (ValueError, SyntaxError):
                expected = exp_str
            sol = Solution()
            method = getattr(sol, method_name, None)
            if method is None:
                q.put(("fail", f"Solution has no method `{method_name}`"))
                return
            result = method(*args)
            if result != expected:
                q.put(("fail",
                       f"test {i}: got {result!r} expected {expected!r}"))
                return
        q.put(("pass", ""))
    except Exception as exc:
        q.put(("fail", f"{type(exc).__name__}: {exc}"))

def score_lcb_problem(code: str, tests: list, method_name: str,
                      timeout: float = 10.0) -> tuple[bool, str]:
    """Sandbox the LCB scoring exec in a child process with a timeout."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_score_lcb_worker, args=(code, tests, method_name, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        if p.is_alive():
            p.kill()
        return False, f"timeout>{timeout}s"
    if q.empty():
        return False, "child crashed (no result)"
    status, msg = q.get_nowait()
    return status == "pass", msg
