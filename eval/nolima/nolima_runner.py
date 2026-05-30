#!/usr/bin/env python3
"""NoLiMa runner against a /v1/chat/completions endpoint (vLLM or llama-server).

Pattern mirrors `eval/lcb/lcb_llama_server.py`: parse CLI args, load needles +
haystack from the `amodaresi/NoLiMa` HF dataset, materialize the (row × test ×
depth × shift) grid, iterate with a sqlite resume cache and a small thread
pool, write per-row samples to `nolima_result.samples.jsonl` + aggregate
metrics to `nolima_result.json`. omk_eval reads that JSON and writes the
canonical `summary.json` on its own.

Why a custom runner rather than wrapping NoLiMa's upstream `run_tests.py`:
the upstream driver is OpenAI-API-shaped but couples ctx-tier configs, prompt
assembly, scoring, and a notebook-only aggregator. We need
sqlite-resume + omk template integration + arbitrary ctx_tokens (no
hardcoded paper tiers) for the 512k YaRN-validation pipeline, and that's
cleaner as a parallel runner that calls our own helpers module.

License: NoLiMa data is Adobe Research, non-commercial research only.
This runner pulls at runtime, never vendors. Scores are derivative output.

Usage:
    python nolima_runner.py \\
        --name MODEL_NAME --base-url http://localhost:8099 \\
        --needle-set needle_set --haystack-tier rand_shuffle \\
        --ctx-tokens 8192 --depth-intervals 26 \\
        --tokenizer /path/to/tokenizer-or-model-dir \\
        --cache-db /path/sqlite.db --output OUT.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
from pathlib import Path

import requests

# Allow `nolima_helpers` from the same directory regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nolima_helpers import (  # noqa: E402
    BookHaystack,
    TokenizerWrapper,
    VALID_METRICS,
    expand_tests,
    load_haystack_text,
    load_needle_set,
    make_cache_key,
    score_response,
)

# Shared sqlite cache (same convention as LCB / MultiPL-E).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from cache_sqlite import SqliteResponseCache  # noqa: E402
except Exception:
    SqliteResponseCache = None


# ── HTTP shim ───────────────────────────────────────────────────────────────

def chat_complete(base_url: str, model: str, system: str, user: str,
                  max_tokens: int, timeout: float) -> dict:
    """Greedy chat completion. Returns {text, prompt_tokens, completion_tokens,
    finish_reason}. Greedy is enforced here at the runner level (independent of
    template) — NoLiMa is a deterministic comprehension test, sampling adds
    noise and breaks cross-cohort comparison.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload,
                      timeout=timeout)
    r.raise_for_status()
    j = r.json()
    choice = j["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    # Same vLLM-reasoning-parser handling as LCB: some builds (Gemma 4 with
    # `--reasoning-parser gemma4`) emit the answer under `reasoning` or
    # `reasoning_content` when the parser fires. NoLiMa expects a short answer
    # (one CHAR name), so either field is fine — take whichever is non-empty.
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    text = content if content else reasoning
    usage = j.get("usage") or {}
    return {
        "text": text,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
    }


# ── Main loop ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True,
                    help="Served model name (passed to /v1/chat/completions)")
    ap.add_argument("--base-url", default="http://localhost:8099")
    ap.add_argument("--needle-set", default="needle_set",
                    help="Stem of the needles JSON in amodaresi/NoLiMa "
                         "(needle_set, needle_set_hard, needle_set_MC, "
                         "needle_set_ONLYDirect, needle_set_w_CoT, "
                         "needle_set_w_Distractor).")
    ap.add_argument("--haystack-tier", default="rand_shuffle",
                    choices=["rand_shuffle", "rand_shuffle_long"],
                    help="rand_shuffle covers up to ~128k ctx; "
                         "rand_shuffle_long for 256k+ extensions.")
    ap.add_argument("--haystack-book", type=int, default=1, choices=[1, 2, 3, 4, 5],
                    help="Which haystack rand_book_N.txt to use (1..5).")
    ap.add_argument("--ctx-tokens", type=int, required=True,
                    help="Target haystack window size in tokens.")
    ap.add_argument("--depth-intervals", type=int, default=26,
                    help="Number of depth-percent placements per (row, test). "
                         "Paper uses 26 below 64k, 11 at 64k+.")
    ap.add_argument("--shifts", type=int, default=1,
                    help="Number of shift-seed rotations of the window (1 = "
                         "deterministic; >1 increases sample density at cost).")
    ap.add_argument("--hop-mode", default="onehop",
                    choices=["onehop", "twohop"])
    ap.add_argument("--tests-per-row", type=int, default=1,
                    help="How many of each needle row's `tests` to expand "
                         "(1 = first only; -1 = all).")
    ap.add_argument("--row-limit", type=int, default=0,
                    help="Cap on number of needle rows used (0 = all).")
    ap.add_argument("--metric", default="contains", choices=list(VALID_METRICS),
                    help="Scoring metric (upstream supports 4 — all string-based, "
                         "case-sensitive, no normalization).")
    ap.add_argument("--tokenizer", required=True,
                    help="Path or HF id of the tokenizer (use the *model's own* "
                         "tokenizer so haystack token counts are accurate).")
    ap.add_argument("--num-concurrent", type=int, default=2,
                    help="Threads in the ThreadPoolExecutor. 2 is safe for "
                         "vLLM long-ctx KV scheduling; higher only helps at "
                         "short ctx with the server idle.")
    ap.add_argument("--max-tokens", type=int, default=192,
                    help="max_tokens cap for the response (upstream uses 192; "
                         "NoLiMa answers are character names — short).")
    ap.add_argument("--http-timeout", type=float, default=900.0)
    ap.add_argument("--cache-db", default=None,
                    help="Sqlite resume DB path (omk_eval always passes this).")
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples-cache", default=None,
                    help="JSONL artifact path; defaults to <output>.samples.jsonl.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore sqlite cache; regenerate every test.")
    ap.add_argument("--early-abort-after", type=int, default=20,
                    help="In-flight sanity: after this many fresh tests, "
                         "abort if all-empty or all-length-cap (infra signal).")
    args = ap.parse_args()

    out_path = Path(args.output)
    cache_path = Path(args.samples_cache) if args.samples_cache else \
        out_path.with_suffix(".samples.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    tests_per_row = None if args.tests_per_row < 0 else args.tests_per_row
    row_limit = None if args.row_limit <= 0 else args.row_limit
    shift_seeds = tuple(range(args.shifts))

    print(f"[nolima] needle_set={args.needle_set} haystack={args.haystack_tier}/"
          f"rand_book_{args.haystack_book}.txt ctx={args.ctx_tokens} "
          f"depths={args.depth_intervals} shifts={args.shifts} "
          f"hop={args.hop_mode} metric={args.metric}", flush=True)

    print("[nolima] loading needles…", flush=True)
    rows = load_needle_set(args.needle_set)
    print(f"[nolima]   {len(rows)} rows", flush=True)

    print("[nolima] loading haystack…", flush=True)
    full_text = load_haystack_text(tier=args.haystack_tier, book_idx=args.haystack_book)
    print(f"[nolima]   {len(full_text):,} chars", flush=True)

    print("[nolima] tokenizing haystack…", flush=True)
    twrap = TokenizerWrapper(args.tokenizer)
    bh = BookHaystack(full_text, encode=twrap.encode, decode=twrap.decode)
    print(f"[nolima]   {bh.token_count:,} tokens, "
          f"{len(bh._snap_points):,} snap points", flush=True)
    if bh.token_count < args.ctx_tokens:
        print(f"[nolima] FATAL: haystack has {bh.token_count} tokens but ctx_tokens="
              f"{args.ctx_tokens}. Switch to --haystack-tier rand_shuffle_long.",
              file=sys.stderr)
        sys.exit(3)

    print("[nolima] expanding test grid…", flush=True)
    tests = expand_tests(
        rows, hop_mode=args.hop_mode, tests_per_row=tests_per_row,
        depth_intervals=args.depth_intervals, ctx_tokens=args.ctx_tokens,
        shift_seeds=shift_seeds, row_limit=row_limit,
    )
    print(f"[nolima]   {len(tests)} tests", flush=True)
    if not tests:
        print("[nolima] FATAL: 0 tests — check --hop-mode and --needle-set.",
              file=sys.stderr)
        sys.exit(2)

    # Resume cache.
    scache = None
    cache_keys: set[str] = set()
    if args.cache_db and SqliteResponseCache is not None:
        scache = SqliteResponseCache(args.cache_db)
        if not args.no_resume:
            cache_keys = set(scache.keys())
    if cache_keys:
        print(f"[nolima] resume: {len(cache_keys)} entries cached", flush=True)

    cache_fp = cache_path.open("a")
    write_lock = threading.Lock()

    fresh_total = 0
    fresh_empty = 0
    fresh_lenhit = 0
    fresh_pass = 0
    cached_pass = 0
    abort_flag = threading.Event()

    def _run_one(t):
        if abort_flag.is_set():
            return None
        key = make_cache_key(t, needle_set=args.needle_set)
        if scache is not None and key in cache_keys:
            rec = scache[key]
            return ("cached", key, t, rec)
        # Build placement (per-test, since shift_seed varies).
        plc = bh.generate(
            context_length=t.ctx_tokens,
            depth_pct=t.depth_pct,
            needle=t.needle_text,
            shift_seed=t.shift_seed,
        )
        user_prompt = t.task_template.format(haystack=plc.text) + "\n" + t.question_text
        gen_t0 = time.time()
        prompt_tokens = completion_tokens = None
        finish_reason = None
        err = ""
        try:
            resp = chat_complete(args.base_url, args.name, t.system_prompt,
                                 user_prompt, args.max_tokens, args.http_timeout)
            completion = resp["text"]
            prompt_tokens = resp["prompt_tokens"]
            completion_tokens = resp["completion_tokens"]
            finish_reason = resp["finish_reason"]
        except Exception as exc:
            completion = ""
            err = f"gen-error: {type(exc).__name__}: {exc}"
        gen_dt = time.time() - gen_t0
        passed = score_response(completion, t.gold_answers, metric=args.metric)
        rec = {
            "doc_id": key,
            "needle_id": t.needle_id,
            "test_id": t.test_id,
            "hop_mode": t.hop_mode,
            "ctx_tokens": t.ctx_tokens,
            "depth_pct": t.depth_pct,
            "shift_seed": t.shift_seed,
            "char_name": t.char_name,
            "needle_text": t.needle_text,
            "question_text": t.question_text,
            "gold_answers": t.gold_answers,
            "completion": completion,
            "passed": bool(passed),
            "metric": args.metric,
            "reason": err if err else ("" if passed else "no_match"),
            "gen_secs": round(gen_dt, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
        }
        # Crash-safe artifact + durable resume store.
        with write_lock:
            cache_fp.write(json.dumps(rec) + "\n")
            cache_fp.flush()
            os.fsync(cache_fp.fileno())
        if scache is not None:
            scache[key] = rec  # autocommit
        return ("fresh", key, t, rec)

    t0 = time.time()
    n_pass = 0
    n_total = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, args.num_concurrent)) as pool:
        for idx, result in enumerate(pool.map(_run_one, tests), start=1):
            if result is None:
                continue
            kind, key, t, rec = result
            n_total += 1
            if rec.get("passed"):
                n_pass += 1
            if kind == "cached":
                cached_pass += 1 if rec.get("passed") else 0
            else:
                fresh_total += 1
                if rec.get("passed"):
                    fresh_pass += 1
                if not (rec.get("completion") or ""):
                    fresh_empty += 1
                if rec.get("finish_reason") == "length":
                    fresh_lenhit += 1
                if (args.early_abort_after
                        and fresh_total >= args.early_abort_after
                        and fresh_pass == 0
                        and (fresh_empty / fresh_total >= 0.8
                             or fresh_lenhit / fresh_total >= 0.8)):
                    print(f"\n[nolima] EARLY ABORT after {fresh_total} fresh "
                          f"tests: pass=0 empty={fresh_empty}/{fresh_total} "
                          f"length_cap={fresh_lenhit}/{fresh_total}", flush=True)
                    abort_flag.set()
            if idx % 10 == 0 or idx == len(tests):
                print(f"[{idx}/{len(tests)}] running={n_pass}/{n_total} "
                      f"({100.0*n_pass/max(1,n_total):.1f}%)", flush=True)

    cache_fp.close()

    elapsed = time.time() - t0
    accuracy = (n_pass / n_total) if n_total else 0.0
    print(f"\n=== {args.name} NoLiMa({args.needle_set}, ctx={args.ctx_tokens}, "
          f"metric={args.metric}): accuracy={accuracy*100:.2f}%  "
          f"({n_pass}/{n_total})  elapsed={elapsed:.0f}s", flush=True)

    # Per-depth breakdown for diagnostic.
    per_depth: dict[str, dict] = {}
    if scache is not None:
        for k in scache.keys():
            r = scache[k]
            if r.get("ctx_tokens") != args.ctx_tokens:
                continue
            d_key = f"{r.get('depth_pct', 0):.4f}"
            slot = per_depth.setdefault(d_key, {"n": 0, "pass": 0})
            slot["n"] += 1
            slot["pass"] += 1 if r.get("passed") else 0

    out = {
        "name": args.name,
        "needle_set": args.needle_set,
        "haystack": f"{args.haystack_tier}/rand_book_{args.haystack_book}.txt",
        "ctx_tokens": args.ctx_tokens,
        "metric": args.metric,
        "hop_mode": args.hop_mode,
        "n": n_total,
        "n_pass": n_pass,
        "accuracy": accuracy,
        "pass_at_1": accuracy,            # surfaced under the canonical key
        "elapsed_secs": elapsed,
        "samples_cache": str(cache_path),
        "per_depth": per_depth,
        "early_abort": abort_flag.is_set(),
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[nolima] wrote {out_path}  (cache: {cache_path})", flush=True)
    if abort_flag.is_set():
        sys.exit(60)


if __name__ == "__main__":
    main()
