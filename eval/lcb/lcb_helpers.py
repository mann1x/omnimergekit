# Auto-extracted LCB helpers from Mythic-RDT/humaneval_smoke.py
from __future__ import annotations

import json
import multiprocessing as mp
import re

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
             max_date: str | None = None,
             testtype: str = "functional",
             task_ids: list[str] | None = None) -> list[dict]:
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
                contest_date = (row.get("contest_date") or "")[:10]
                # task_ids mode: the curated list is the single source of truth
                # — it already encodes difficulty + date window at BUILD time, so
                # bypass the difficulty/date filters here. This lets a frozen
                # subset mix difficulties (e.g. lcb_v6_55 = 44 medium + 11 hard).
                # The structural filters below (functional testtype + class-based
                # starter) STILL apply — the scorer requires them. No-op when
                # task_ids is None (legacy filter-mode is byte-identical).
                if task_ids is None:
                    if row.get("difficulty") != difficulty:
                        continue
                    if contest_date and contest_date < min_date:
                        continue
                    if max_date and contest_date and contest_date >= max_date:
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
                tid = f"lcb/{row.get('platform','?')}/{row.get('question_id','?')}"
                if task_ids is not None and tid not in task_ids:
                    continue
                out.append({
                    "task_id": tid,
                    "question_content": row.get("question_content", ""),
                    "starter_code": starter,
                    "method_name": m.group(1),
                    "difficulty": row.get("difficulty", ""),
                    "contest_date": contest_date,
                    "public_tests": public,
                })
                if task_ids is None and len(out) >= limit:
                    break
        if task_ids is None and len(out) >= limit:
            break
    if task_ids is not None:
        order = {t: i for i, t in enumerate(task_ids)}
        out.sort(key=lambda r: order.get(r["task_id"], 1_000_000))
    print(f"[lcb] loaded {len(out)} problems "
          f"(difficulty={difficulty}, post {min_date}, functional"
          f"{', task_ids=' + str(len(task_ids)) if task_ids else ''})")
    return out

def clean_lcb_completion(completion: str, starter_code: str) -> str:
    """Extract the Solution class — official LCB semantics: last fence pair.

    Official LCB (lcb_runner/utils/extraction_utils.py:extract_code) splits by
    line, locates all lines containing ```, and returns the slice between the
    LAST TWO fence-containing lines. We do the same to avoid penalizing
    models (Gemma 4 in particular) that emit a first-attempt then a corrected
    rewrite — first-fence-wins systematically picks the wrong block.
    """
    lines = completion.split("\n")
    idx = [i for i, line in enumerate(lines) if "```" in line]
    if len(idx) >= 2:
        return "\n".join(lines[idx[-2] + 1 : idx[-1]])
    # bug-606 (2026-08-20): exactly ONE fence line -- the OPENER of an unterminated
    # block, which is what a runaway/cap-truncated generation looks like. The old
    # fallback stripped only a TRAILING fence, so that opening ```python survived into
    # the scored code and made line 1 a SyntaxError by construction. Drop the opener
    # too; everything after it is the (possibly truncated) block.
    #
    # MEASURED OUTCOME of the offline rescore over all 24 banked LCB cells
    # (ream_arms lcb_v6_77q + _48k, qwen_suite lcb_v6_77q; 674 failures total):
    #   single-fence failures      48   <- the population this branch touches
    #   recleaned by the fix       48
    #   now ast.parse (was not)    37
    #   flipped fail -> pass        0
    #   cells whose score moved     0
    # 48/48 of them carry finish_reason="length". Every single-fence case is a
    # cap-truncated runaway, so the retained opener was masking a TRUNCATION, not
    # hiding a working solution: fixing it turns "SyntaxError line 1" into the honest
    # "got None expected 6" / IndentationError, and recovers nothing. NO ARM'S LCB
    # SCORE CHANGES. Do not restate the earlier 27/224, 32/254, 10/196 figures as this
    # fix's blast radius -- those counted the broader "cleaned does not parse but some
    # fenced block in raw does" class, which is dominated by MULTI-fence replies
    # (538/674 failures; 27 damaged) where the official last-two-fence slice picks a
    # non-parsing block though an earlier one parses. That is a SEPARATE defect this
    # branch does not address.
    #
    # Corollary worth keeping: an LCB failure reported as a line-1 SyntaxError was, in
    # this cohort, usually a mislabelled truncation -- our own extractor was converting
    # length-capped generations into syntax errors and hiding them from any truncation
    # census (cf. bug-592).
    if len(idx) == 1:
        i = idx[0]
        after = "\n".join(lines[i + 1:])
        before = "\n".join(lines[:i])
        # The opener is normally the first non-empty line; if there is real code
        # BEFORE it, the single fence is a closer instead -- keep what precedes it.
        body = after if not before.strip() else before
        return TRAILING_FENCE_RE.sub("", body)
    # No fence at all; strip a trailing partial fence and return as-is.
    return TRAILING_FENCE_RE.sub("", completion)

def _parse_io(s: str):
    """Parse an LCB input/output string with official LCB semantics:
    json.loads first (handles JSON true/false/null and lists/dicts/numbers),
    fall back to ast.literal_eval for legacy Python-literal-encoded values,
    and finally return the raw string when neither parses."""
    import ast as _ast
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return _ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def _score_lcb_worker(code: str, tests: list, method_name: str, q):
    """Run inside child process. Imports are inside the function so they
    survive the fork in subprocesses without re-importing module globals."""
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
            # LCB encodes IO as JSON (official semantics: json.loads). Two
            # input formats observed:
            #   - Single-line: the WHOLE string is one arg (e.g. "[1,2,3]"
            #     means one List[int] arg).
            #   - Multi-line: each line is one positional arg.
            if "\n" in inp_str:
                args = []
                for line in inp_str.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    args.append(_parse_io(line))
                args = tuple(args)
            else:
                args = (_parse_io(inp_str),)
            expected = _parse_io(exp_str)
            sol = Solution()
            method = getattr(sol, method_name, None)
            if method is None:
                q.put(("fail", f"Solution has no method `{method_name}`"))
                return
            result = method(*args)
            # Tuple→list coercion (official LCB: "don't penalize models for
            # returning tuples where the ground truth is a list").
            if isinstance(result, tuple):
                result = list(result)
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
