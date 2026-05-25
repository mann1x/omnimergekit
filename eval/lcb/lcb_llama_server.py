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

# Shared sqlite response cache (eval/cache_sqlite.py, one dir up). Optional:
# when present + --cache-db given, sqlite is the durable resume store
# (2026-05-23 "all evals resume through sqlite" directive). Degrades to the
# legacy jsonl resume if the module is unavailable (e.g. a stripped pod).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from cache_sqlite import SqliteResponseCache  # noqa: E402
except Exception:
    SqliteResponseCache = None


def chat_complete(base_url: str, model: str, prompt: str, max_tokens: int,
                  timeout: float = 600.0,
                  thinking_budget: int | None = None,
                  enable_thinking: bool | None = None) -> dict:
    """Returns dict {text, prompt_tokens, completion_tokens, finish_reason}.

    `finish_reason="length"` is the cap-hit fingerprint — that response was
    truncated mid-generation and is unlikely to score correctly. The caller
    should propagate this so audits can distinguish capability from truncation.

    When `thinking_budget` is set and the vLLM server is configured with
    `--reasoning-parser gemma4` (or similar), vLLM force-transitions the
    model from thinking to answer phase at that token count. Without this,
    Gemma 4 instruct can think for the full `max_tokens` window and never
    emit a parseable answer (parser sees an unclosed thinking block and
    drops both content+reasoning → 75% empty FAIL rate observed
    2026-05-13 on 128e LCB-55).
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    # vLLM /v1/chat/completions accepts extra SamplingParams as TOP-LEVEL
    # JSON fields (matching how the OpenAI Python SDK auto-forwards unknown
    # kwargs via extra_body — they end up at the JSON root). Match that
    # contract: thinking_token_budget at root, enable_thinking inside
    # chat_template_kwargs (it's a template flag, not a sampling param).
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {
            "enable_thinking": bool(enable_thinking),
        }
    if thinking_budget is not None and thinking_budget > 0:
        payload["thinking_token_budget"] = int(thinking_budget)
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
    # vLLM emits the reasoning trace under `reasoning` (newer builds) or
    # `reasoning_content` (older / llama-server / some configs). Read both
    # and prefer whichever is non-empty. Validated 2026-05-13 on vLLM 0.1.dev16519
    # with --reasoning-parser gemma4 → field name is `reasoning`.
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
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
    ap.add_argument("--task-ids", default="",
                    help="Comma-separated explicit task_id list (e.g. "
                         "'lcb/leetcode/3579,lcb/leetcode/3566'). When set, "
                         "only these problems are loaded (post-filter); "
                         "--limit caps the final size. Output order matches "
                         "the list. Used by smoke templates that pin a "
                         "curated subset rather than 'first N by date'.")
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
    ap.add_argument("--cache-db", default=None,
                    help="Sqlite resume DB path (omk_eval passes "
                         "<out_dir>/sqlite_cache/<prefix>_<tag>.db). When set, "
                         "sqlite is the durable resume store keyed by task_id; "
                         "the .samples.jsonl is still written as the scoring "
                         "artifact. Omit for standalone jsonl-only resume.")
    ap.add_argument("--thinking-budget", type=int, default=0,
                    help="vLLM thinking_token_budget cap (0=disabled). Mandatory "
                         "for Gemma 4 + --reasoning-parser gemma4 — without it "
                         "the model can think past max_tokens and the parser "
                         "drops both content+reasoning (empty FAIL).")
    ap.add_argument("--enable-thinking", default="",
                    help="Forward chat_template_kwargs.enable_thinking "
                         "('true'/'false'/empty for unset).")
    ap.add_argument("--early-abort-after", type=int, default=5,
                    help="In-flight sanity: after this many fresh problems "
                         "(uncached), abort if pass rate is 0 AND >=80%% "
                         "have chars=0 (empty response) OR finish_reason=length. "
                         "Saves hours when thinking budget / parser is misconfigured. "
                         "Set 0 to disable.")
    ap.add_argument("--http-timeout", type=float, default=900.0,
                    help="Per-request HTTP read timeout (seconds). vLLM "
                         "NVFP4A16 Gemma 4 generates at ~22 tok/s on a "
                         "3090 — with max_tokens=16384 a single long "
                         "reasoning trace can take ~760s. Was 600 (too "
                         "tight for the slow quant); now 900 default. "
                         "Set higher when seeing recurring ReadTimeouts.")
    args = ap.parse_args()

    out_path = Path(args.output)
    cache_path = Path(args.samples_cache) if args.samples_cache else \
        out_path.with_suffix(".samples.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    tid_list = [t.strip() for t in args.task_ids.split(",") if t.strip()] \
        if args.task_ids else None
    problems = load_lcb(limit=args.limit, difficulty=args.difficulty,
                        min_date=args.min_date, testtype="functional",
                        task_ids=tid_list)
    if not problems:
        print("[lcb] no problems loaded; aborting", file=sys.stderr)
        sys.exit(2)

    # Resume store. Directive (2026-05-23): all evals resume through sqlite.
    # When --cache-db is given (omk_eval always passes it) sqlite is the durable
    # source of truth; the .samples.jsonl is still written as the scoring /
    # inspection artifact. Without --cache-db (bare CLI / legacy pod) fall back
    # to the jsonl resume so the runner still works standalone.
    scache = None
    if args.cache_db and SqliteResponseCache is not None:
        scache = SqliteResponseCache(args.cache_db)
        cache = {} if args.no_resume else {k: scache[k] for k in scache.keys()}
        store = "sqlite"
    else:
        cache = {} if args.no_resume else load_cache(cache_path)
        store = "jsonl"
    if cache:
        print(f"[lcb] resume: {len(cache)} problem(s) already cached ({store})")

    cache_fp = cache_path.open("a")  # append mode preserves prior cache lines (artifact)

    n_pass = 0
    per_problem = []
    t0 = time.time()
    # In-flight sanity counters — fresh = uncached problems generated this run.
    fresh_total = 0
    fresh_pass = 0
    fresh_empty = 0
    fresh_lenhit = 0
    early_abort = False

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
            et = None
            if args.enable_thinking.lower() in ("true", "1", "yes"):
                et = True
            elif args.enable_thinking.lower() in ("false", "0", "no"):
                et = False
            resp = chat_complete(args.base_url, args.name, prompt,
                                 args.max_tokens,
                                 timeout=args.http_timeout,
                                 thinking_budget=args.thinking_budget or None,
                                 enable_thinking=et)
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
            "doc_id": tid,          # uniform with lm-eval samples; token_stats dedups on this
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
        os.fsync(cache_fp.fileno())  # crash-safe artifact
        if scache is not None:
            scache[tid] = rec        # durable resume store (sqlitedict autocommit)

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

        # In-flight sanity: count this fresh (uncached) problem's outcome.
        # When `fresh_total` reaches `--early-abort-after`, abort if the
        # signature looks like an infra failure (all empty / all length-cap)
        # rather than letting the chain burn hours on a broken config.
        fresh_total += 1
        if passed:
            fresh_pass += 1
        if len(completion) == 0:
            fresh_empty += 1
        if finish_reason == "length":
            fresh_lenhit += 1
        if (args.early_abort_after and fresh_total >= args.early_abort_after
                and fresh_pass == 0
                and (fresh_empty / fresh_total >= 0.8
                     or fresh_lenhit / fresh_total >= 0.8)):
            print(f"\n[lcb] EARLY ABORT after {fresh_total} fresh problems:\n"
                  f"  pass=0   empty={fresh_empty}/{fresh_total}"
                  f"   length_cap={fresh_lenhit}/{fresh_total}\n"
                  f"  → likely infra/parser/budget bug, not a model bug.\n"
                  f"  Check: --thinking-budget, --enable-thinking,\n"
                  f"  vLLM --reasoning-parser, response field name (reasoning\n"
                  f"  vs reasoning_content).", flush=True)
            early_abort = True
            break

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
    # Distinct exit code for early abort so omk_eval / the chain script can
    # surface it as "infra failure" (vs "model failure") in summaries.
    if early_abort:
        sys.exit(60)


if __name__ == "__main__":
    main()
