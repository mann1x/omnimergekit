#!/usr/bin/env python3
"""LCB-Medium runner against a llama-server /v1/chat/completions endpoint.

Reuses LCB loading + scoring from Mythic-RDT/scripts/humaneval_smoke.py
(same dataset filter, same sandboxed scorer). Generation is delegated to
llama-server so we can eval GGUF quants without loading transformers.

Usage:
    python lcb_llama_server.py --name MODEL_NAME --tokenizer DIR \
        --base-url http://localhost:8099 --limit 30 --output OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

# Make humaneval_smoke importable
MYTHIC_SCRIPTS = "/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/Mythic-RDT/scripts"
sys.path.insert(0, MYTHIC_SCRIPTS)
from humaneval_smoke import (  # noqa: E402
    LCB_INSTRUCT_TEMPLATE,
    clean_lcb_completion,
    load_lcb,
    score_lcb_problem,
)


def chat_complete(base_url: str, model: str, prompt: str, max_tokens: int,
                  timeout: float = 600.0) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    r = requests.post(f"{base_url}/v1/chat/completions",
                      json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    msg = j["choices"][0]["message"]
    # Some llama-server builds (Qwen3.5 chat template) push the entire
    # generation into `reasoning_content` and leave `content` empty. Take
    # whichever is non-empty; if both are populated, concatenate them so
    # clean_lcb_completion can extract a fenced code block from either.
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    if content and reasoning:
        return reasoning + "\n" + content
    return content or reasoning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-url", default="http://localhost:8099")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--difficulty", default="medium")
    ap.add_argument("--min-date", default="2024-10-01")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--output", required=True,
                    help="JSON output path with results + per-problem")
    ap.add_argument("--samples", default=None,
                    help="Optional JSONL path for raw samples")
    args = ap.parse_args()

    problems = load_lcb(limit=args.limit, difficulty=args.difficulty,
                        min_date=args.min_date, testtype="functional")
    if not problems:
        print("[lcb] no problems loaded; aborting", file=sys.stderr)
        sys.exit(2)

    samples_fp = open(args.samples, "w") if args.samples else None

    n_pass = 0
    per_problem = []
    t0 = time.time()
    for i, prob in enumerate(problems):
        prompt = LCB_INSTRUCT_TEMPLATE.format(
            question=prob["question_content"],
            starter=prob["starter_code"],
        )
        gen_t0 = time.time()
        try:
            completion = chat_complete(args.base_url, args.name, prompt,
                                       args.max_tokens)
        except Exception as exc:
            completion = ""
            err = f"gen-error: {type(exc).__name__}: {exc}"
            print(f"[{i+1}/{len(problems)}] {prob['task_id']} {err}",
                  file=sys.stderr)
        else:
            err = ""
        gen_dt = time.time() - gen_t0

        cleaned = clean_lcb_completion(completion, prob["starter_code"])
        passed, reason = score_lcb_problem(
            cleaned, prob["public_tests"], prob["method_name"],
        )
        if passed:
            n_pass += 1
        per_problem.append({
            "task_id": prob["task_id"],
            "passed": bool(passed),
            "reason": reason if not passed else "",
            "gen_secs": round(gen_dt, 2),
            "completion_chars": len(completion),
        })
        print(f"[{i+1}/{len(problems)}] {prob['task_id']} "
              f"{'PASS' if passed else 'FAIL'} {gen_dt:.1f}s  "
              f"running={n_pass}/{i+1}",
              flush=True)
        if samples_fp:
            samples_fp.write(json.dumps({
                "task_id": prob["task_id"],
                "completion": completion,
                "cleaned": cleaned,
                "passed": passed,
                "reason": reason,
            }) + "\n")
            samples_fp.flush()

    if samples_fp:
        samples_fp.close()

    pass_at_1 = n_pass / len(problems)
    elapsed = time.time() - t0
    print(f"\n=== {args.name} LCB-{args.difficulty} ({len(problems)}q): "
          f"pass@1 = {pass_at_1*100:.2f}%  ({n_pass}/{len(problems)})  "
          f"elapsed={elapsed:.0f}s")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "name": args.name,
            "difficulty": args.difficulty,
            "min_date": args.min_date,
            "n": len(problems),
            "n_pass": n_pass,
            "pass_at_1": pass_at_1,
            "elapsed_secs": elapsed,
            "per_problem": per_problem,
        }, f, indent=2)
    print(f"[lcb] wrote {args.output}")


if __name__ == "__main__":
    main()
