#!/usr/bin/env python3
"""
rescore_strip_think.py — strip <think>...</think> from HE/MBPP samples and rescore.

Qwen3.6-Omnimerge-v4 is a reasoning model. Raw /v1/completions returns
<think>...</think> + code. lm_eval's HE/MBPP scorers do `exec(prompt + completion + tests)`
which SyntaxErrors on the literal `<` in `<think>`. We strip the think block before re-exec.

Usage:
    python3 rescore_strip_think.py --bench humaneval --samples <path> --tests humaneval
    python3 rescore_strip_think.py --bench mbpp --samples <path> --tests mbpp

Emits:
    {samples_dir}/rescored_clean.json  — { "pass@1": float, "n": int, "n_pass": int,
                                            "n_syntax_err": int, "n_assertion_fail": int,
                                            "n_other_err": int }

The think-strip is non-greedy across newlines: re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
"""
import argparse
import json
import re
import sys
import signal
from pathlib import Path


THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# Some reasoning models wrap final code in ```python fences. Strip those.
FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks. If no closing </think>, drop everything
    up to the first ```python (HE) or first `def ` / `import ` line as a fallback."""
    if "<think>" in text and "</think>" in text:
        return THINK_RE.sub("", text, count=1)
    if "<think>" in text:
        # No close tag — try to find the first python-code marker
        for marker in ("```python", "```\n", "\ndef ", "\nimport ", "\nfrom "):
            i = text.find(marker)
            if i >= 0:
                return text[i:].lstrip("`")
        # Last resort: drop everything before first newline-after-think
        i = text.find("</")
        if i >= 0:
            return text[i + 2:]
    return text


def extract_code(text: str) -> str:
    """After think-strip, also peel a code fence if present."""
    m = FENCE_RE.search(text)
    if m:
        return m.group(1).strip() + "\n"
    return text


def _run_one(args):
    """Worker: exec(prompt+completion+tests) under timeout. Returns (status, error_type)."""
    prompt, completion, tests, timeout = args
    code = prompt + completion + "\n" + tests
    # Run in a subprocess to isolate
    def handler(signum, frame):
        raise TimeoutError("exec timeout")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        ns: dict = {}
        exec(code, ns)
    except SyntaxError as e:
        return ("syntax_err", repr(e)[:100])
    except AssertionError as e:
        return ("assertion_fail", repr(e)[:100])
    except TimeoutError:
        return ("timeout", "timeout")
    except Exception as e:
        return ("other_err", repr(type(e).__name__) + ": " + repr(e)[:100])
    else:
        return ("pass", "")
    finally:
        signal.alarm(0)


def rescore_humaneval(samples_path: Path, timeout: int = 10):
    """HE: each sample has `arguments[0]` = prompt; `target` = canonical; `resps[0][0]` = completion;
    test cases are in `doc.test` (HE) or `doc.text` (MBPP)."""
    results = {"pass": 0, "syntax_err": 0, "assertion_fail": 0, "timeout": 0, "other_err": 0}
    n = 0
    failures = []
    with open(samples_path) as f:
        for line in f:
            sample = json.loads(line)
            n += 1
            doc = sample.get("doc", {})
            prompt = doc.get("prompt", "")
            tests = doc.get("test", "")
            entry_point = doc.get("entry_point", "")
            resp = sample.get("resps", [[None]])[0][0] or sample.get("filtered_resps", [None])[0] or ""
            completion_raw = resp
            completion = extract_code(strip_think(completion_raw))
            # HE assertion format
            full_tests = tests + f"\ncheck({entry_point})\n"
            status, errdetail = _run_one((prompt, completion, full_tests, timeout))
            results[status] = results.get(status, 0) + 1
            if status != "pass":
                failures.append({"task_id": doc.get("task_id"), "status": status,
                                 "error": errdetail[:200],
                                 "comp_head": completion[:150]})
    return n, results, failures


def rescore_mbpp(samples_path: Path, timeout: int = 10):
    """MBPP: doc.text=problem, doc.test_list=list of assert strings, doc.test_setup_code (optional)."""
    results = {"pass": 0, "syntax_err": 0, "assertion_fail": 0, "timeout": 0, "other_err": 0}
    n = 0
    failures = []
    with open(samples_path) as f:
        for line in f:
            sample = json.loads(line)
            n += 1
            doc = sample.get("doc", {})
            tests_list = doc.get("test_list", []) or []
            setup = doc.get("test_setup_code", "") or ""
            resp = sample.get("resps", [[None]])[0][0] or sample.get("filtered_resps", [None])[0] or ""
            completion = extract_code(strip_think(resp))
            tests_block = setup + "\n" + "\n".join(tests_list) + "\n"
            # MBPP doesn't pass a prompt to exec — the completion IS the function def.
            status, errdetail = _run_one(("", completion, tests_block, timeout))
            results[status] = results.get(status, 0) + 1
            if status != "pass":
                failures.append({"task_id": doc.get("task_id"), "status": status,
                                 "error": errdetail[:200],
                                 "comp_head": completion[:150]})
    return n, results, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["humaneval", "mbpp"])
    ap.add_argument("--samples", required=True, help="path to samples_*.jsonl")
    ap.add_argument("--output", default=None, help="output JSON (default: <samples_dir>/rescored_clean.json)")
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--max-failures-shown", type=int, default=10)
    args = ap.parse_args()

    samples_path = Path(args.samples)
    if not samples_path.exists():
        print(f"ERROR: samples file not found: {samples_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else samples_path.parent / "rescored_clean.json"

    if args.bench == "humaneval":
        n, results, failures = rescore_humaneval(samples_path, args.timeout)
    else:
        n, results, failures = rescore_mbpp(samples_path, args.timeout)

    n_pass = results.get("pass", 0)
    pass_at_1 = n_pass / n if n > 0 else 0.0

    summary = {
        "bench": args.bench,
        "n": n,
        "n_pass": n_pass,
        "pass@1": pass_at_1,
        "n_syntax_err": results.get("syntax_err", 0),
        "n_assertion_fail": results.get("assertion_fail", 0),
        "n_timeout": results.get("timeout", 0),
        "n_other_err": results.get("other_err", 0),
        "samples_path": str(samples_path),
        "failures_sample": failures[:args.max_failures_shown],
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"=== rescore {args.bench} ===")
    print(f"  n: {n}")
    print(f"  pass@1: {pass_at_1:.4f}  ({n_pass}/{n})")
    print(f"  syntax_err: {results.get('syntax_err', 0)}")
    print(f"  assertion_fail: {results.get('assertion_fail', 0)}")
    print(f"  timeout: {results.get('timeout', 0)}")
    print(f"  other_err: {results.get('other_err', 0)}")
    print(f"  written: {out_path}")


if __name__ == "__main__":
    main()
