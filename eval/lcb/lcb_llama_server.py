#!/usr/bin/env python3
"""LCB-Medium runner against a llama-server /v1/chat/completions endpoint.

Reuses LCB loading + scoring from a local lcb_helpers shim (extracted from
Mythic-RDT/humaneval_smoke.py — see lcb_helpers.py in this directory). The
shim is dependency-free (no torch, no transformers, no mythic_rdt) so the
runner can also live on a fresh pod.

Generation is delegated to llama-server so we can eval GGUF quants without
loading transformers. Per-problem cache (JSONL) preserves the full
generation + cleaned code + reason for every problem and makes the runner
**resumable on crash**. The aggregate JSON contains both summary fields and
the full per-problem gen text (no information dropped).

Usage:
    python lcb_llama_server.py --name MODEL_NAME \\
        --base-url http://localhost:8099 \\
        --limit 999 --output OUT.json

The cache is written next to OUT.json as OUT.samples.jsonl unless
--samples-cache is overridden. Re-running with the same --output and the
same problem set skips problems already present in the cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

# Allow `lcb_helpers` to be imported from the same directory as this script,
# regardless of where it's launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from lcb_helpers import (  # noqa: E402
        LCB_INSTRUCT_TEMPLATE,
        clean_lcb_completion,
        load_lcb,
        score_lcb_problem,
    )
except ImportError:
    # Fallback: Mythic-RDT working tree on solidpc (legacy path)
    MYTHIC_SCRIPTS = "/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/Mythic-RDT/scripts"
    sys.path.insert(0, MYTHIC_SCRIPTS)
    from humaneval_smoke import (  # noqa: E402
        LCB_INSTRUCT_TEMPLATE,
        clean_lcb_completion,
        load_lcb,
        score_lcb_problem,
    )


def chat_complete(base_url: str, model: str, prompt: str, max_tokens: int,
                  timeout: float = 600.0) -> dict:
    """Returns dict {text, prompt_tokens, completion_tokens, finish_reason}.

    `finish_reason="length"` is the cap-hit fingerprint — that response was
    truncated mid-generation and is unlikely to score correctly. The caller
    should propagate this so audits can distinguish capability from truncation.
    """
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
    choice = j["choices"][0]
    msg = choice["message"]
    # Some llama-server builds (Qwen3.5 chat template) push the entire
    # generation into `reasoning_content` and leave `content` empty. Take
    # whichever is non-empty; if both are populated, concatenate them so
    # clean_lcb_completion can extract a fenced code block from either.
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    if content and reasoning:
        text = reasoning + "\n" + content
    else:
        text = content or reasoning
    usage = j.get("usage") or {}
    return {
        "text": text,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
    }


def load_cache(path: Path) -> dict:
    """Read the JSONL cache and return a dict task_id → record."""
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = rec.get("task_id")
            if tid:
                out[tid] = rec
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-url", default="http://localhost:8099")
    ap.add_argument("--limit", type=int, default=30,
                    help="Max problems (use 999 for full medium ~55q)")
    ap.add_argument("--difficulty", default="medium")
    ap.add_argument("--min-date", default="2024-10-01")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="Server max_tokens cap. Was 2048; bumped to 8192 "
                         "after observing pod p90 gen ~2000 tokens hitting "
                         "the cap and producing truncated output that "
                         "fails to parse. Pair with server -c >= 32768.")
    ap.add_argument("--output", required=True,
                    help="JSON output path with results + per-problem (incl. full generations)")
    ap.add_argument("--samples-cache", default=None,
                    help="JSONL cache path; defaults to <output>.samples.jsonl. "
                         "Re-runs skip problems already cached here.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore existing cache; regenerate every problem.")
    args = ap.parse_args()

    out_path = Path(args.output)
    cache_path = Path(args.samples_cache) if args.samples_cache else \
        out_path.with_suffix(".samples.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    problems = load_lcb(limit=args.limit, difficulty=args.difficulty,
                        min_date=args.min_date, testtype="functional")
    if not problems:
        print("[lcb] no problems loaded; aborting", file=sys.stderr)
        sys.exit(2)

    cache = {} if args.no_resume else load_cache(cache_path)
    if cache:
        print(f"[lcb] resume: {len(cache)} problem(s) already cached at {cache_path}")

    cache_fp = cache_path.open("a")  # append mode preserves prior cache lines

    n_pass = 0
    per_problem = []
    t0 = time.time()

    for i, prob in enumerate(problems):
        tid = prob["task_id"]
        if tid in cache:
            rec = cache[tid]
            n_pass += 1 if rec.get("passed") else 0
            per_problem.append({
                "task_id": tid,
                "passed": bool(rec.get("passed")),
                "reason": rec.get("reason", "") if not rec.get("passed") else "",
                "gen_secs": rec.get("gen_secs", 0.0),
                "completion_chars": len(rec.get("completion", "") or ""),
                "completion": rec.get("completion", ""),
                "cleaned": rec.get("cleaned", ""),
            })
            print(f"[{i+1}/{len(problems)}] {tid} CACHED "
                  f"({'PASS' if rec.get('passed') else 'FAIL'})  "
                  f"running={n_pass}/{i+1}", flush=True)
            continue

        prompt = LCB_INSTRUCT_TEMPLATE.format(
            question=prob["question_content"],
            starter=prob["starter_code"],
        )
        gen_t0 = time.time()
        prompt_tokens = completion_tokens = None
        finish_reason = None
        try:
            resp = chat_complete(args.base_url, args.name, prompt,
                                 args.max_tokens)
            completion = resp["text"]
            prompt_tokens = resp["prompt_tokens"]
            completion_tokens = resp["completion_tokens"]
            finish_reason = resp["finish_reason"]
            err = ""
        except Exception as exc:
            completion = ""
            err = f"gen-error: {type(exc).__name__}: {exc}"
            print(f"[{i+1}/{len(problems)}] {tid} {err}", file=sys.stderr)
        gen_dt = time.time() - gen_t0

        cleaned = clean_lcb_completion(completion, prob["starter_code"])
        passed, reason = score_lcb_problem(
            cleaned, prob["public_tests"], prob["method_name"],
        )
        if err and not reason:
            reason = err
        if passed:
            n_pass += 1

        rec = {
            "task_id": tid,
            "passed": bool(passed),
            "reason": reason if not passed else "",
            "gen_secs": round(gen_dt, 2),
            "completion": completion,
            "cleaned": cleaned,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
        }
        cache_fp.write(json.dumps(rec) + "\n")
        cache_fp.flush()
        os.fsync(cache_fp.fileno())  # crash-safe: don't lose results on SIGKILL

        per_problem.append({
            "task_id": tid,
            "passed": bool(passed),
            "reason": reason if not passed else "",
            "gen_secs": round(gen_dt, 2),
            "completion_chars": len(completion),
            "completion": completion,
            "cleaned": cleaned,
        })
        print(f"[{i+1}/{len(problems)}] {tid} "
              f"{'PASS' if passed else 'FAIL'} {gen_dt:.1f}s  "
              f"chars={len(completion)}  running={n_pass}/{i+1}",
              flush=True)

    cache_fp.close()

    pass_at_1 = n_pass / len(problems)
    elapsed = time.time() - t0
    print(f"\n=== {args.name} LCB-{args.difficulty} ({len(problems)}q): "
          f"pass@1 = {pass_at_1*100:.2f}%  ({n_pass}/{len(problems)})  "
          f"elapsed={elapsed:.0f}s")

    with out_path.open("w") as f:
        json.dump({
            "name": args.name,
            "difficulty": args.difficulty,
            "min_date": args.min_date,
            "n": len(problems),
            "n_pass": n_pass,
            "pass_at_1": pass_at_1,
            "elapsed_secs": elapsed,
            "samples_cache": str(cache_path),
            "per_problem": per_problem,
        }, f, indent=2)
    print(f"[lcb] wrote {out_path}  (cache: {cache_path})")


if __name__ == "__main__":
    main()
