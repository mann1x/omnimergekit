#!/usr/bin/env python3
"""MultiPL-E generation phase (omk_eval backend).

Reads the per-language MultiPL-E HumanEval split from `nuprl/MultiPL-E`, hits a
running llama-server `/v1/completions` endpoint with GREEDY decoding (temp=0)
and writes one JSON file per problem in the schema the
`ghcr.io/nuprl/multipl-e-evaluation` Docker image expects:

    {
      "name": "<problem_name>",
      "language": "<lang>",
      "prompt":   "<original prompt>",
      "completions": ["<single greedy completion>"],   # list, len=1 for pass@1
      "tests":    "<test snippet>"
    }

Phase 2 (`multipl_e_evaluate.sh` → docker run) executes the language-specific
compiler/runtime and reports pass@1.

Resume (2026-05-23 "all evals through sqlite" directive): the durable resume
store is a sqlite DB (`--cache-db`, keyed `f"{lang}::{name}"`). On a cache hit
the per-problem JSON is re-materialized from the cached completion WITHOUT an
HTTP call, so the Docker eval still sees a complete `--out-dir`. The per-problem
`.json` existence is a secondary skip. Without `--cache-db` we fall back to the
JSON-file resume so the script still runs standalone.

Usage:
    multipl_e_generate.py --lang rs \\
        --base-url http://localhost:8099/v1/completions \\
        --out-dir <WS>/multipl_e/generations/<NAME>/humaneval-rs \\
        --cache-db <out>/sqlite_cache/mpe_100_<tag>.db \\
        [--max-tokens 1024] [--limit 100] [--concurrency 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import load_dataset

# Shared sqlite response cache (eval/cache_sqlite.py, two dirs up). Optional:
# falls back to JSON-file resume when unavailable (e.g. a stripped pod).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from cache_sqlite import SqliteResponseCache
except Exception:
    SqliteResponseCache = None

# Portable HF datasets cache: honor the environment (pods set HF_HOME /
# HF_DATASETS_CACHE under /workspace; solidpc under backup_models). Never /tmp.
DEFAULT_HF_CACHE = (
    os.environ.get("HF_DATASETS_CACHE")
    or (os.environ.get("HF_HOME") and str(Path(os.environ["HF_HOME"]) / "datasets"))
    or str(Path.home() / ".cache" / "huggingface" / "datasets")
)


def already_generated(out_path: Path) -> bool:
    if not out_path.exists():
        return False
    try:
        d = json.loads(out_path.read_text())
        comps = d.get("completions") or []
        return len(comps) > 0 and isinstance(comps[0], str) and len(comps[0]) > 0
    except Exception:
        return False


def make_request(base_url: str, prompt: str, stop: list[str], max_tokens: int,
                 model_name: str, timeout: int = 600,
                 max_retries: int = 6) -> str:
    """Greedy completion with retry on 5xx/transient errors. Returns generated
    text only. Exponential backoff (4/8/16/32/64/128 s) on HTTP 5xx,
    ConnectionError, Timeout, or malformed/empty JSON. A clean 4xx raises
    immediately. Server errors are NEVER silently dropped."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "stop": stop,
        "stream": False,
    }
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(base_url, json=payload, timeout=timeout)
            if r.status_code >= 500:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                raise requests.HTTPError(last_err, response=r)
            r.raise_for_status()
            j = r.json()
            if "choices" not in j or not j["choices"]:
                raise ValueError(f"malformed response: {str(j)[:200]}")
            return j["choices"][0]["text"]
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError,
                ValueError) as e:
            last_err = str(e)
            if attempt >= max_retries:
                break
            backoff = 4 * (2 ** attempt)
            print(f"    [retry {attempt+1}/{max_retries}] {last_err[:120]} "
                  f"— sleeping {backoff}s", flush=True)
            time.sleep(backoff)
    raise RuntimeError(f"giving up after {max_retries} retries: {last_err}")


def _write_problem_json(out_path: Path, name: str, lang: str, prompt: str,
                        completion: str, tests: str, stop: list[str]) -> None:
    payload = {
        "name": name,
        "language": lang,
        "prompt": prompt,
        "completions": [completion],
        "tests": tests,
        "stop_tokens": stop,
    }
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(out_path)


def gen_one(args, doc, out_dir: Path, scache, lock) -> tuple[str, str, float, int]:
    name = doc["name"]
    out_path = out_dir / f"{name}.json"
    key = f"{args.lang}::{name}"

    prompt = doc["prompt"]
    stop = list(doc.get("stop_tokens") or [])
    if "<file_sep>" not in stop:
        stop.append("<file_sep>")
    tests = doc["tests"]

    # 1) sqlite cache hit → re-materialize JSON (Docker eval needs it), no HTTP.
    if scache is not None:
        with lock:
            cached = scache.get(key) if key in scache else None
        if cached and cached.get("completion"):
            if not already_generated(out_path):
                _write_problem_json(out_path, name, doc["language"], prompt,
                                    cached["completion"], tests, stop)
            return name, "cached(sqlite)", 0.0, len(cached["completion"])

    # 2) JSON-file resume (standalone / no sqlite).
    if already_generated(out_path):
        return name, "cached(json)", 0.0, 0

    t0 = time.time()
    completion = make_request(
        args.base_url, prompt, stop[:4],  # /v1/completions caps stop at 4
        args.max_tokens, args.model_name)
    elapsed = time.time() - t0

    _write_problem_json(out_path, name, doc["language"], prompt, completion, tests, stop)
    if scache is not None:
        with lock:
            scache[key] = {"completion": completion, "name": name,
                           "language": doc["language"], "gen_secs": round(elapsed, 2)}
    return name, "ok", elapsed, len(completion)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, help="Language code: rs, java, js, …")
    ap.add_argument("--base-url", required=True, help="llama-server /v1/completions URL")
    ap.add_argument("--model-name", default="multipl-e",
                    help="OpenAI 'model' field (server ignores it)")
    ap.add_argument("--out-dir", required=True, help="Per-problem JSON output dir")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0,
                    help="First N problems (0 = full split)")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="Parallel requests (match server --parallel)")
    ap.add_argument("--cache-db", default=None,
                    help="Sqlite resume DB (keyed lang::name). Durable resume "
                         "store per the all-evals-through-sqlite rule.")
    ap.add_argument("--hf-cache", default=DEFAULT_HF_CACHE,
                    help="HF datasets cache root (never /tmp)")
    args = ap.parse_args()

    os.environ["HF_DATASETS_CACHE"] = args.hf_cache
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scache = None
    lock = threading.Lock()
    if args.cache_db and SqliteResponseCache is not None:
        scache = SqliteResponseCache(args.cache_db)

    cfg = f"humaneval-{args.lang}"
    print(f"[gen] loading dataset nuprl/MultiPL-E config={cfg} cache={args.hf_cache}",
          flush=True)
    ds = load_dataset("nuprl/MultiPL-E", cfg, split="test")
    docs = list(ds)
    if args.limit > 0:
        docs = docs[: args.limit]
    print(f"[gen] {cfg}: {len(docs)} problems  out_dir={out_dir}", flush=True)

    started = time.time()
    n_done = n_err = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(gen_one, args, d, out_dir, scache, lock): d["name"]
                   for d in docs}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                name, status, elapsed, n_chars = fut.result()
                n_done += 1
                if n_done % 10 == 0 or n_done == len(docs):
                    pace = (time.time() - started) / max(n_done, 1)
                    eta = pace * (len(docs) - n_done)
                    print(f"  [{n_done}/{len(docs)}] {name}: {status} "
                          f"({elapsed:.1f}s, {n_chars} chars; "
                          f"pace {pace:.1f}s/q, ETA {eta/60:.1f}m)", flush=True)
            except Exception as exc:
                n_err += 1
                print(f"  [ERR] {name}: {exc}", flush=True)

    if scache is not None:
        scache.close()
    print(f"[gen] done: ok={n_done - n_err} err={n_err} "
          f"total_elapsed={(time.time() - started)/60:.1f}m  out_dir={out_dir}",
          flush=True)
    if n_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
