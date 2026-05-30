#!/usr/bin/env python3
"""RULER runner against a /v1/completions endpoint (vLLM or llama-server).

Three sequential phases per (task, max_seq_length, num_samples):

  1. **prepare** — subprocess into upstream `<RULER_ROOT>/scripts/data/prepare.py`
     to generate `<stage>/data/<task>/validation.jsonl`. Idempotent — re-running
     with the same args is a no-op when the file already exists. This is the
     only piece of upstream RULER code we exec; everything else is ours.

  2. **infer** — own /v1/completions (raw text continuation) loop with sqlite
     resume (`eval/cache_sqlite.py:SqliteResponseCache`) and a small
     ThreadPoolExecutor. Greedy (temp=0, top_p=1) is enforced at the runner
     level regardless of template; RULER VT/NIAH/CWE/FWE are deterministic
     substring tasks — sampling adds noise and breaks cross-cohort comparison.

     Why /v1/completions and not /v1/chat/completions: `prepare.py` already
     bakes the chosen `model_template_type` (base / meta-chat / Phi3 / …)
     AND the per-task `answer_prefix` into the staged `input` field — the
     model is meant to continue raw text from there. Re-wrapping that in a
     chat template would make the served instruct-tuned model see the
     trailing "Answer: …" prefix as a fragment-ending user message and
     reformulate the question instead of completing it. lm-eval's bundled
     RULER tasks use `local-completions` for the same reason.

  3. **score** — in-process `ruler_helpers.score_task` (verbatim port of
     upstream `string_match_all` from `scripts/eval/synthetic/constants.py:25`).
     See ruler_helpers.py header for the inline-vs-subprocess RCA.

Outputs:
  - <output>.json                — aggregate score + provenance
  - <output>.samples.jsonl       — one record per sample (resume-safe append)
  - <stage>/data/<task>/         — upstream staged inputs (regenerable)

CLI matches the NoLiMa runner so omk_eval.dispatch_ruler_native can wire
template fields → args without translation.

License: NVIDIA/RULER is Apache-2.0. The runtime clone is the canonical source
(`/workspace/RULER` on pods, `/shared/dev/RULER` on solidpc, or `$RULER_ROOT`).
This runner pulls at runtime, never vendors. Score artifacts are derivative
output, freely publishable per standard benchmark conventions.
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

# Allow `ruler_helpers` from the same directory regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ruler_helpers import (  # noqa: E402
    ensure_nltk_data,
    load_validation_jsonl,
    locate_ruler_root,
    make_cache_key,
    metric_for_task,
    run_prepare,
    score_task,
)

# Shared sqlite cache (same convention as LCB / MultiPL-E / NoLiMa).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from cache_sqlite import SqliteResponseCache  # noqa: E402
except Exception:
    SqliteResponseCache = None


# ── HTTP shim ───────────────────────────────────────────────────────────────

def text_complete(base_url: str, model: str, prompt: str,
                  max_tokens: int, timeout: float,
                  system_prompt: str | None = None) -> dict:
    """Greedy text completion via /v1/completions. Returns
    {text, prompt_tokens, completion_tokens, finish_reason}.

    Why /v1/completions and not /v1/chat/completions:
        Upstream `prepare.py` already bakes the chosen `model_template_type`
        (base / meta-chat / Phi3 / meta-llama3 / …) into the staged `input`
        AND appends the per-task `answer_prefix` to it
        (RULER/scripts/data/prepare.py:101:
            config['template'] = model_template.format(...) + answer_prefix
        ). Sending that already-wrapped string back through a chat API would
        re-wrap it in the served model's chat template, which means the
        served model sees the prefix `Answer: According to …, they are: ` as
        a fragment-ending user message and reformulates the question instead
        of continuing it. lm-eval's bundled RULER tasks use `local-completions`
        for the same reason.

        Empirically validated 2026-05-28: chat-completions on Gemma 4 26B-A4B-it
        Q6_K against the 5-sample VT smoke produced bullet-formatted
        reformulations like "*   Goal: Find all variables …" and scored 0/5;
        switching to /v1/completions makes the model continue the prefix and
        emit the variable names directly.

        `system_prompt` is ignored on /v1/completions (raw mode has no roles);
        the argument is kept for CLI compatibility but is a no-op.
    """
    _ = system_prompt  # explicit no-op: /v1/completions has no roles
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    r = requests.post(f"{base_url}/v1/completions", json=payload,
                      timeout=timeout)
    r.raise_for_status()
    j = r.json()
    choice = j["choices"][0]
    text = choice.get("text") or ""
    usage = j.get("usage") or {}
    return {
        "text": text,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--name", required=True,
                    help="Served model name (passed to /v1/completions)")
    ap.add_argument("--base-url", default="http://localhost:8099")
    ap.add_argument("--task", required=True,
                    help="RULER task (vt, cwe, fwe, qa_1, qa_2, niah_single_*, "
                         "niah_multikey_*, niah_multiquery, niah_multivalue). "
                         "Refused at startup if not in ruler_helpers.TASK_METRICS.")
    ap.add_argument("--ctx-tokens", type=int, required=True,
                    help="max_seq_length passed to upstream prepare.py.")
    ap.add_argument("--num-samples", type=int, default=50,
                    help="num_samples passed to upstream prepare.py.")
    ap.add_argument("--tokenizer", required=True,
                    help="Path or HF id of the tokenizer for prepare.py "
                         "(use the *model's own* tokenizer so token counts in "
                         "the staged inputs match what the served model sees).")
    ap.add_argument("--tokenizer-type", default="hf",
                    choices=("hf", "nemo", "openai"),
                    help="Tokenizer type. We use HF AutoTokenizer; nemo/openai "
                         "exist for upstream parity but are not exercised here.")
    ap.add_argument("--model-template-type", default="base",
                    help="upstream `model_template_type` (template.py family). "
                         "'base' = no chat scaffolding; the served model's own "
                         "chat template handles message formatting. Don't switch "
                         "without re-baselining against published RULER numbers.")
    ap.add_argument("--ruler-root", default=None,
                    help="Override RULER clone path. Default: $RULER_ROOT → "
                         "/workspace/RULER → /shared/dev/RULER.")
    ap.add_argument("--stage-dir", required=True,
                    help="Directory for upstream-generated validation.jsonl. "
                         "Re-runs reuse the stage; delete to force regen.")
    ap.add_argument("--num-concurrent", type=int, default=2,
                    help="ThreadPoolExecutor size. 2 is safe for vLLM long-ctx "
                         "KV scheduling; raise only at short ctx with the "
                         "server idle.")
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="max_tokens cap for the response. RULER answers are "
                         "always short (variable names, counts, single words) — "
                         "32-128 is plenty; higher just wastes ctx.")
    ap.add_argument("--http-timeout", type=float, default=1200.0,
                    help="HTTP timeout (s). 20 min is the safe ceiling for 512k "
                         "ctx on vLLM NVFP4A16; override via template "
                         "backend_overrides for shorter ctx.")
    ap.add_argument("--system-prompt", default=None,
                    help="Optional system message. RULER's base template does "
                         "not require one; leave unset unless you've verified "
                         "your served model needs it.")
    ap.add_argument("--cache-db", default=None,
                    help="Sqlite resume DB path (omk_eval always passes this).")
    ap.add_argument("--output", required=True,
                    help="Path to ruler_result.json")
    ap.add_argument("--samples-cache", default=None,
                    help="JSONL artifact path; defaults to <output>.samples.jsonl.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore sqlite cache; regenerate every sample.")
    ap.add_argument("--early-abort-after", type=int, default=20,
                    help="In-flight sanity: after N fresh samples, abort if "
                         "all-empty / all-length-cap (server-side failure signal). "
                         "NOTE: length-cap rate ~50% on Gemma 4 instruct VT is "
                         "normal (~30-tok answer-prefix preamble + 50-tok names + "
                         "rumination); the 0.8 abort threshold accommodates that.")
    ap.add_argument("--random-seed", type=int, default=42,
                    help="Passed through to prepare.py for deterministic input "
                         "generation. RULER's published numbers use 42.")
    args = ap.parse_args()

    out_path = Path(args.output)
    cache_path = Path(args.samples_cache) if args.samples_cache else \
        out_path.with_suffix(".samples.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Reject unknown RULER tasks BEFORE we touch prepare.py — fail loud now,
    # not after a multi-minute prepare run.
    try:
        metric_name = metric_for_task(args.task)
    except KeyError as e:
        print(f"[ruler] FATAL: {e}", file=sys.stderr)
        sys.exit(2)

    ruler_root = Path(args.ruler_root) if args.ruler_root else locate_ruler_root()
    if not (ruler_root / "scripts" / "data" / "prepare.py").is_file():
        print(f"[ruler] FATAL: ruler_root={ruler_root} does not contain "
              f"scripts/data/prepare.py", file=sys.stderr)
        sys.exit(2)

    print(f"[ruler] task={args.task}  ctx={args.ctx_tokens}  "
          f"n={args.num_samples}  metric={metric_name}  root={ruler_root}",
          flush=True)

    ensure_nltk_data()

    # ── PHASE 1: prepare ────────────────────────────────────────────────────
    # Pass --save_dir <stage>/data (NOT <stage>/data/<task>); upstream
    # prepare.py:111 writes to <save_dir>/<task>/<subset>.jsonl on its own.
    stage_data_dir = Path(args.stage_dir) / "data"
    print(f"[ruler] phase 1: prepare → {stage_data_dir}/{args.task}/", flush=True)
    val_jsonl = run_prepare(
        ruler_root=ruler_root,
        task=args.task,
        max_seq_length=args.ctx_tokens,
        num_samples=args.num_samples,
        tokenizer_path=args.tokenizer,
        tokenizer_type=args.tokenizer_type,
        save_dir=stage_data_dir,
        model_template_type=args.model_template_type,
        random_seed=args.random_seed,
    )
    # For downstream summary reporting, the "task stage dir" is the per-task
    # subdir upstream chose, not the parent.
    stage_dir = stage_data_dir / args.task
    rows = load_validation_jsonl(val_jsonl)
    if not rows:
        print(f"[ruler] FATAL: prepare produced 0 rows at {val_jsonl}",
              file=sys.stderr)
        sys.exit(2)
    print(f"[ruler]   {len(rows)} samples staged", flush=True)

    # ── PHASE 2: infer ──────────────────────────────────────────────────────
    print(f"[ruler] phase 2: infer (concurrency={args.num_concurrent}, "
          f"max_tokens={args.max_tokens})", flush=True)

    scache = None
    cache_keys: set[str] = set()
    if args.cache_db and SqliteResponseCache is not None:
        scache = SqliteResponseCache(args.cache_db)
        if not args.no_resume:
            cache_keys = set(scache.keys())
    if cache_keys:
        print(f"[ruler]   resume: {len(cache_keys)} entries cached", flush=True)

    cache_fp = cache_path.open("a")
    write_lock = threading.Lock()

    fresh_total = 0
    fresh_empty = 0
    fresh_lenhit = 0
    abort_flag = threading.Event()

    def _run_one(row):
        if abort_flag.is_set():
            return None
        key = make_cache_key(task=args.task, max_seq_length=args.ctx_tokens,
                             sample_index=int(row["index"]))
        if scache is not None and key in cache_keys:
            return ("cached", key, row, scache[key])
        gen_t0 = time.time()
        err = ""
        completion = ""
        prompt_tokens = completion_tokens = None
        finish_reason = None
        try:
            # Upstream RULER prepare.py keeps `input` and `answer_prefix` as
            # separate JSONL fields — our original docstring (line 19/86-89)
            # incorrectly assumed prepare.py would fold answer_prefix INTO
            # input. lm-eval's ruler tasks concatenate them via niah_single_1
            # .yaml's `gen_prefix: "{{gen_prefix}}"` interpolation; without
            # this concat, the served model has to invent the answer preamble
            # itself, eating most of max_gen_toks and producing a wildly low
            # score (T143.3 Phase 2 ship-gate: 0.00% vs lm-eval's 94.80% on
            # the SAME model + SAME max_gen_toks=30 at vt/32k). Fix: pre-pend
            # answer_prefix exactly as lm-eval does.
            full_prompt = row["input"] + row.get("answer_prefix", "")
            resp = text_complete(args.base_url, args.name,
                                 prompt=full_prompt,
                                 max_tokens=args.max_tokens,
                                 timeout=args.http_timeout,
                                 system_prompt=args.system_prompt)
            completion = resp["text"]
            prompt_tokens = resp["prompt_tokens"]
            completion_tokens = resp["completion_tokens"]
            finish_reason = resp["finish_reason"]
        except Exception as exc:
            err = f"gen-error: {type(exc).__name__}: {exc}"
        gen_dt = time.time() - gen_t0
        rec = {
            "doc_id": key,
            "task": args.task,
            "ctx_tokens": args.ctx_tokens,
            "sample_index": int(row["index"]),
            "input_preview": (row["input"][:200] + "…") if len(row["input"]) > 200 else row["input"],
            "input_length_chars": len(row["input"]),
            "input_length_tokens": row.get("length"),
            "outputs": row["outputs"],
            "completion": completion,
            "gen_secs": round(gen_dt, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "reason": err,
        }
        with write_lock:
            cache_fp.write(json.dumps(rec) + "\n")
            cache_fp.flush()
            os.fsync(cache_fp.fileno())
        if scache is not None:
            scache[key] = rec
        return ("fresh", key, row, rec)

    t0 = time.time()
    n_total = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, args.num_concurrent)) as pool:
        for idx, result in enumerate(pool.map(_run_one, rows), start=1):
            if result is None:
                continue
            kind, _key, _row, rec = result
            n_total += 1
            if kind == "fresh":
                fresh_total += 1
                if not (rec.get("completion") or ""):
                    fresh_empty += 1
                if rec.get("finish_reason") == "length":
                    fresh_lenhit += 1
                if (args.early_abort_after
                        and fresh_total >= args.early_abort_after
                        and (fresh_empty / fresh_total >= 0.8
                             or fresh_lenhit / fresh_total >= 0.8)):
                    print(f"\n[ruler] EARLY ABORT after {fresh_total} fresh "
                          f"samples: empty={fresh_empty}/{fresh_total} "
                          f"length_cap={fresh_lenhit}/{fresh_total} — "
                          f"check server (silent-empty or token starvation).",
                          flush=True)
                    abort_flag.set()
            if idx % 10 == 0 or idx == len(rows):
                print(f"[ruler]   [{idx}/{len(rows)}] elapsed={time.time()-t0:.0f}s",
                      flush=True)

    cache_fp.close()
    elapsed = time.time() - t0

    # ── PHASE 3: score ──────────────────────────────────────────────────────
    print("[ruler] phase 3: score", flush=True)
    # Collect predictions in index order (deterministic). For resumed samples
    # the cache is the source of truth; for in-this-run samples, we use the
    # records we just wrote. Re-read from scache (or samples_cache jsonl when
    # scache is unavailable) so resumed-only runs work.
    pred_by_idx: dict[int, str] = {}
    if scache is not None:
        for k in scache.keys():
            rec = scache[k]
            if (rec.get("task") == args.task
                    and rec.get("ctx_tokens") == args.ctx_tokens):
                pred_by_idx[int(rec["sample_index"])] = rec.get("completion") or ""
    else:
        # Reread samples.jsonl (small file; this is only the fallback path).
        with cache_path.open() as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                rec = json.loads(ln)
                pred_by_idx[int(rec["sample_index"])] = rec.get("completion") or ""

    preds: list[str] = []
    refs: list[list[str]] = []
    missing: list[int] = []
    for row in rows:
        idx = int(row["index"])
        if idx not in pred_by_idx:
            missing.append(idx)
            continue
        preds.append(pred_by_idx[idx])
        refs.append(list(row["outputs"]))

    if missing:
        print(f"[ruler] WARN: {len(missing)} samples missing predictions "
              f"(first few: {missing[:5]}) — scoring only complete samples.",
              flush=True)

    score = score_task(args.task, preds, refs) if preds else 0.0

    print(f"\n=== RULER {args.task} @ ctx={args.ctx_tokens}: {score:.2f}%  "
          f"(n={len(preds)}, metric={metric_name})  elapsed={elapsed:.0f}s",
          flush=True)

    out = {
        "name": args.name,
        "task": args.task,
        "ctx_tokens": args.ctx_tokens,
        "num_samples": args.num_samples,
        "metric": metric_name,             # "string_match_all" / "string_match_part"
        "n": len(preds),
        "missing": len(missing),
        "score": float(score),             # 0-100 scale (RULER convention)
        "pass_at_1": float(score) / 100.0, # 0-1 canonical for omk roll-up
        "accuracy": float(score) / 100.0,  # alias
        "elapsed_secs": elapsed,
        "fresh_total": fresh_total,
        "fresh_empty": fresh_empty,
        "fresh_lenhit": fresh_lenhit,
        "samples_cache": str(cache_path),
        "stage_dir": str(stage_dir),
        "early_abort": abort_flag.is_set(),
        "ruler_root": str(ruler_root),
        "model_template_type": args.model_template_type,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[ruler] wrote {out_path}  (cache: {cache_path})", flush=True)
    if abort_flag.is_set():
        sys.exit(60)


if __name__ == "__main__":
    main()
