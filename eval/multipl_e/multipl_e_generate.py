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
import re
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
                 max_retries: int = 6,
                 temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int = 0) -> str:
    """Completion (defaults greedy) with retry on 5xx/transient errors. Returns
    text only. Exponential backoff (4/8/16/32/64/128 s) on HTTP 5xx,
    ConnectionError, Timeout, or malformed/empty JSON. A clean 4xx raises
    immediately. Server errors are NEVER silently dropped."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
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


# --- chat mode -------------------------------------------------------------
# Reasoning/instruct Gemma-4 degenerates on raw /v1/completions (runs to the
# token cap, never emits the column-0 stop terminator). The fix is to generate
# via /v1/chat/completions with the chat template (same path that gives HE+
# ~90%), extract the fenced code, then convert the model's full function back
# into a *body-only* completion so MultiPL-E's `prompt + completion + tests`
# assembly stays valid (prompt carries imports/class/sig; tests supply closers).

_CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)


# bug-604 taxonomy gate: problems whose extracted body lost an opening line
# (brace balance < 0 on a language where the tests supply the closing braces).
# Appended by chat_to_body(), reported as a count by main(). Never silently empty
# a cell — surface it so a guaranteed-zero problem is not read as a model failure.
UNBALANCED_BODIES: list[str] = []


def extract_code_block(text: str) -> str:
    """Return the largest fenced code block; fall back to the whole text."""
    blocks = _CODE_FENCE_RE.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip("\n")
    return (text or "").strip()


def chat_to_body(prompt: str, code: str, stop_tokens: list[str],
                 name: str = "") -> str:
    """Convert a chat model's full-function reply into a body-only completion
    that drops into MultiPL-E's `prompt + completion + tests` assembly.

    Anchor on the prompt's signature line (its last non-blank line, ending with
    '{'); take everything after it. If the tests supply the closing brace(s)
    (the dataset stop token is a bare '}', true for rs/java), strip the trailing
    run of closing braces so we don't double-close — this handles both
    single-level (rust: 1 brace) and nested (java: method + class: 2 braces).
    For js (stop tokens are `\\nfunction `/comment markers, not a bare brace)
    the completion self-closes, so the trailing brace is kept.

    bug-604 (2026-08-20): when the reply does NOT restate the signature the old
    fallback did `code[code.find("{") + 1:]`, which for a BODY-ONLY reply eats
    the first loop/if opening line and orphans its '}'. Measured across all 9
    arms of multipl_e_100: 176/900 humaneval-java completions, 176 of 176 FATAL.
    A body-only reply is already the body — return it untouched.
    """
    plines = [ln for ln in prompt.splitlines() if ln.strip()]
    anchor = plines[-1] if plines else ""
    idx = code.find(anchor) if anchor else -1
    if idx < 0 and anchor:
        # whitespace-tolerant match on the signature minus the trailing '{'
        target = anchor.rstrip().rstrip("{").strip()
        for line in code.splitlines():
            if target and target in line:
                anchor, idx = line, code.find(line)
                break
    if idx >= 0:
        after = code[idx + len(anchor):]
        body = after if anchor.rstrip().endswith("{") else (
            after[after.find("{") + 1:] if "{" in after else after)
    else:
        # bug-604: neither the exact nor the whitespace-tolerant anchor matched, so
        # the reply never restated the signature => `code` IS ALREADY the body.
        # Do NOT strip to the first '{': in a body-only reply that brace belongs to
        # the first loop/if, and eating its line orphans the matching '}'. The class
        # then closes early, MultiPL-E's appended main() lands outside it, and javac
        # reports "illegal start of type". Was: code[code.find("{") + 1:].
        body = code

    tests_supply_close = any((t or "").strip() == "}" for t in (stop_tokens or []))
    if tests_supply_close:
        body = re.sub(r"\s*(?:\}\s*)+\Z", "\n", body)
        # TAXONOMY GATE (bug-604). When the tests supply the closing braces a
        # well-formed body is brace-BALANCED; a NEGATIVE balance means an opening
        # line was lost and the assembled program cannot compile. Emit it (never
        # silently drop a cell) but record it so the run reports a count instead of
        # banking guaranteed-zero problems as if they were model failures.
        # This check is only valid inside this branch: for js the stop tokens are
        # not a bare '}', tests_supply_close is False, and a negative balance is the
        # NORMAL shape (measured 895/900 completions, only 74 of them failing).
        if body.count("{") - body.count("}") < 0:
            UNBALANCED_BODIES.append(name or "<unnamed>")
    return body if body.endswith("\n") else body + "\n"


JAVA_PROMPT = (
    "import java.util.*;\nclass Problem {\n"
    "    // Return a greatest common divisor of two integers a and b\n"
    "    public static long greatestCommonDivisor(long a, long b) {\n")
JAVA_BODY = ("        while (b != 0) {\n            long temp = b;\n"
             "            b = a % b;\n            a = temp;\n        }\n"
             "        return a;\n")
JAVA_STOP = ["\n    }\n", "<file_sep>"]


def selftest() -> int:
    """Golds for chat_to_body. bug-604: the body-only fallback used to eat the
    first loop's opening line. Gold 3 is the exact historical broken output —
    it must never be produced again."""
    fails = []
    total = 0

    def chk(tag, got, want):
        nonlocal total
        total += 1
        ok = got == want
        print(f"  [{'PASS' if ok else 'FAIL'}] {tag}")
        if not ok:
            print(f"          got  {got!r}\n          want {want!r}")
            fails.append(tag)

    # 1) BODY-ONLY reply (the bug-604 shape): body survives intact, balanced.
    got = chat_to_body(JAVA_PROMPT, JAVA_BODY, JAVA_STOP, "gold_body_only")
    chk("java body-only keeps the `while` opening line", "while (b != 0) {" in got, True)
    chk("java body-only is brace-balanced", got.count("{") - got.count("}"), 0)

    # 2) FULL-FUNCTION reply: anchor path, same body out.
    full = ("class Problem {\n"
            "    public static long greatestCommonDivisor(long a, long b) {\n"
            + JAVA_BODY + "    }\n}\n")
    got2 = chat_to_body(JAVA_PROMPT, full, JAVA_STOP, "gold_full_fn")
    chk("java full-function keeps the `while` opening line", "while (b != 0) {" in got2, True)
    chk("java full-function is brace-balanced", got2.count("{") - got2.count("}"), 0)

    # 3) REGRESSION GOLD: the exact string the old fallback produced.
    broken = ("\n            long temp = b;\n            b = a % b;\n"
              "            a = temp;\n        }\n        return a;\n")
    chk("historical broken output is NOT reproduced", got == broken, False)

    # 4) The taxonomy gate fires on a genuinely unbalanced body...
    before = len(UNBALANCED_BODIES)
    chat_to_body(JAVA_PROMPT, "        foo();\n        }\n        return a;\n",
                 JAVA_STOP, "gold_unbalanced")
    chk("gate flags a negative-balance java body", len(UNBALANCED_BODIES) > before, True)

    # 5) ...and stays silent for js, where negative balance is the NORMAL shape
    #    (stop tokens are not a bare '}', so tests_supply_close is False).
    before = len(UNBALANCED_BODIES)
    chat_to_body("function f(a) {\n", "  return a;\n}\n",
                 ["\nfunction ", "\n//"], "gold_js")
    chk("gate stays silent for js negative balance", len(UNBALANCED_BODIES), before)

    print(f"\nSELFTEST {'OK' if not fails else 'FAILED: ' + ', '.join(fails)} "
          f"({total - len(fails)}/{total})")
    return 1 if fails else 0


def make_chat_request(chat_url: str, prompt: str, lang: str, max_tokens: int,
                      model_name: str, timeout: int = 600,
                      max_retries: int = 6,
                      temperature: float = 0.0, top_p: float = 1.0,
                      top_k: int = 0) -> str:
    """Chat completion (defaults greedy). Returns the assistant content (code
    extracted by the caller). Same retry/backoff policy as make_request."""
    instruction = (
        f"Complete the following {lang} function. Reply with ONLY the complete "
        f"function implementation in a single Markdown code block — include the "
        f"signature exactly as given, write the full body, and do NOT add any "
        f"explanation, example usage, or test code.\n\n```{lang}\n{prompt}\n```"
    )
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": instruction}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": False,
    }
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(chat_url, json=payload, timeout=timeout)
            if r.status_code >= 500:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                raise requests.HTTPError(last_err, response=r)
            r.raise_for_status()
            j = r.json()
            if "choices" not in j or not j["choices"]:
                raise ValueError(f"malformed response: {str(j)[:200]}")
            msg = j["choices"][0].get("message", {}) or {}
            # content first; fall back to reasoning_content only if content empty
            return msg.get("content") or msg.get("reasoning_content") or ""
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
    if args.mode == "chat":
        chat_url = args.base_url.replace("/v1/completions", "/v1/chat/completions")
        raw = make_chat_request(chat_url, prompt, args.lang,
                                args.max_tokens, args.model_name,
                                temperature=args.temperature, top_p=args.top_p,
                                top_k=args.top_k)
        completion = chat_to_body(prompt, extract_code_block(raw), stop, name)
    else:
        completion = make_request(
            args.base_url, prompt, stop[:4],  # /v1/completions caps stop at 4
            args.max_tokens, args.model_name,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    elapsed = time.time() - t0

    _write_problem_json(out_path, name, doc["language"], prompt, completion, tests, stop)
    if scache is not None:
        with lock:
            scache[key] = {"completion": completion, "name": name,
                           "language": doc["language"], "gen_secs": round(elapsed, 2)}
    return name, "ok", elapsed, len(completion)


def main():
    # bug-604 golds run standalone, before the required serving args are parsed:
    # `python multipl_e_generate.py --selftest` must work with no server, no GPU.
    if "--selftest" in sys.argv[1:]:
        sys.exit(selftest())

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="Run chat_to_body golds (bug-604) and exit; no server needed")
    ap.add_argument("--lang", required=True, help="Language code: rs, java, js, …")
    ap.add_argument("--base-url", required=True, help="llama-server /v1/completions URL")
    ap.add_argument("--model-name", default="multipl-e",
                    help="OpenAI 'model' field (server ignores it)")
    ap.add_argument("--out-dir", required=True, help="Per-problem JSON output dir")
    ap.add_argument("--mode", choices=("completion", "chat"), default="completion",
                    help="completion = raw /v1/completions (base-style); chat = "
                         "/v1/chat/completions + chat template + code extraction "
                         "(required for reasoning/instruct models like Gemma-4)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (0 = greedy; default greedy for "
                         "frozen canonical MPE). Shadow templates set the gemma "
                         "vendor sampler via omk_eval --metadata generation.*")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="First N problems (0 = full split)")
    ap.add_argument("--problems", default="",
                    help="Comma-separated problem-name allowlist (e.g. "
                         "HumanEval_65_circular_shift,HumanEval_89_encrypt). When "
                         "set, ONLY these problems are generated and --limit is "
                         "ignored. Used by the 21q rumination per-problem screen.")
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
    if args.problems:
        keep = {p.strip() for p in args.problems.split(",") if p.strip()}
        docs = [d for d in docs if d["name"] in keep]
        missing = keep - {d["name"] for d in docs}
        if missing:
            print(f"[gen] WARN: {len(missing)} requested problems not found in {cfg}: "
                  f"{sorted(missing)}", flush=True)
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

    # bug-604 taxonomy gate. A body whose braces close more than they open on a
    # language where the tests supply the closing braces cannot compile — it is a
    # GUARANTEED zero, and read as a model failure it silently deflates the score.
    # Report it as a rate so a future regression is visible in the log, not only
    # in a post-hoc join. (list.append from the worker threads is GIL-atomic.)
    if UNBALANCED_BODIES:
        n_ub = len(UNBALANCED_BODIES)
        print(f"[gen] WARN bug-604: {n_ub}/{len(docs)} completions have negative "
              f"brace balance and CANNOT compile — these are guaranteed zeros, not "
              f"model failures. First 10: {sorted(UNBALANCED_BODIES)[:10]}",
              flush=True)
    else:
        print(f"[gen] bug-604 gate: 0/{len(docs)} unbalanced bodies", flush=True)

    if n_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
