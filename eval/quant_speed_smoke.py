#!/usr/bin/env python3
"""Throughput smoke for a vLLM-served quant.

Drives 3 reasoning-heavy prompts at temperature=0, records per-prompt
prompt_tokens / completion_tokens / elapsed, prints a single-line summary
with median tok/s. Used to compare quant runtimes on the same model
before committing 1-2 wall-hours to a full LCB-55 run.

The 3 prompts are LCB-medium-flavored (algorithm + reasoning) at 4096
max_tokens — long enough that the per-token kernel cost dominates over
warmup. Same prompts for every quant, so the only varying factor is the
backend throughput.

Usage:
    quant_speed_smoke.py --url http://localhost:8195 --name <served> \\
        [--max-tokens 4096] [--n 3]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import requests


PROMPTS = [
    # Algorithmic — moderate generation length, deterministic
    "Write a Python function `is_prime(n: int) -> bool` that returns True iff n is prime. "
    "Then call it on 1..30 and print the result.",
    # Reasoning + code — typically 2-4k tokens of trace
    "Solve: given a sorted array of integers, return the indices of the two numbers that "
    "sum to a given target. Return a Python Solution class with a `twoSum(self, nums, target)` "
    "method. Explain your approach first, then provide the code.",
    # Longer generation — closer to LCB-medium worst case
    "Implement a Python class `LRUCache` with `get(key) -> int` and `put(key, value)`. "
    "Both O(1). Walk through your design choices (linked list + dict), then provide the full "
    "implementation with type hints and a brief test in `if __name__ == '__main__'`.",
]


def time_one(url: str, name: str, prompt: str, max_tokens: int,
             timeout: float) -> dict:
    payload = {
        "model": name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.time()
    try:
        r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "elapsed": time.time() - t0}
    elapsed = time.time() - t0
    usage = j.get("usage", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    fr = j["choices"][0].get("finish_reason", "?")
    tps = ct / elapsed if elapsed > 0 else 0
    return {
        "elapsed": elapsed,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "finish_reason": fr,
        "tok_per_s": tps,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8195")
    ap.add_argument("--name", required=True, help="served-model-name")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--n", type=int, default=3, choices=(1, 2, 3),
                    help="how many prompts to time (1-3)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--json", action="store_true", help="emit single-line JSON")
    args = ap.parse_args()

    prompts = PROMPTS[:args.n]
    rows = []
    for i, p in enumerate(prompts):
        if not args.json:
            print(f"[{i+1}/{len(prompts)}] timing... ({len(p)} chars in prompt)",
                  file=sys.stderr)
        rows.append(time_one(args.url, args.name, p, args.max_tokens, args.timeout))

    tps = [r["tok_per_s"] for r in rows if "error" not in r and r["completion_tokens"] > 0]
    cts = [r["completion_tokens"] for r in rows if "error" not in r]
    elapsed_total = sum(r.get("elapsed", 0) for r in rows)
    summary = {
        "name": args.name,
        "n": len(rows),
        "errors": [r["error"] for r in rows if "error" in r],
        "median_tok_per_s": statistics.median(tps) if tps else None,
        "mean_tok_per_s": (sum(tps) / len(tps)) if tps else None,
        "max_tok_per_s": max(tps) if tps else None,
        "min_tok_per_s": min(tps) if tps else None,
        "median_completion_tokens": statistics.median(cts) if cts else None,
        "elapsed_total_s": round(elapsed_total, 2),
        "per_prompt": rows,
    }

    if args.json:
        print(json.dumps(summary))
    else:
        print()
        print(f"=== quant speed smoke: {args.name} ===")
        print(f"  n prompts:        {summary['n']} (max_tokens={args.max_tokens})")
        if summary["errors"]:
            print(f"  errors:           {summary['errors']}")
        if tps:
            print(f"  median tok/s:     {summary['median_tok_per_s']:.2f}")
            print(f"  mean tok/s:       {summary['mean_tok_per_s']:.2f}")
            print(f"  range tok/s:      {summary['min_tok_per_s']:.2f} – {summary['max_tok_per_s']:.2f}")
            print(f"  median completion tokens: {summary['median_completion_tokens']:.0f}")
        print(f"  total elapsed:    {summary['elapsed_total_s']}s")
    return 0 if tps else 3


if __name__ == "__main__":
    sys.exit(main())
