#!/usr/bin/env python3
"""OpenAI MRCR runner against a /v1/chat/completions endpoint (vLLM or llama-server).

Three sequential phases per (bin, needles, num_samples):

  1. **prepare** — download the needle parquets from openai/mrcr (HF cache, never
     vendored), o200k-bin to the requested context band, and select up to
     `num_samples` rows (deterministic, seed-shuffled, round-robin balanced across
     the 2/4/8 needle counts). See mrcr_helpers.iter_bin_samples.

  2. **infer** — own /v1/chat/completions loop with sqlite resume
     (eval/cache_sqlite.py:SqliteResponseCache) and a small ThreadPoolExecutor.
     The MRCR `prompt` IS the multi-turn message list — we send it verbatim.
     Greedy (temp=0, top_p=1) is enforced at the runner level: MRCR's
     SequenceMatcher metric is deterministic; sampling adds noise and breaks
     cross-cohort comparison (the frozen-greedy canonical rule).

     THINKING MUST BE OFF. MRCR grades on `response.startswith(random_hash)`; a
     leading <think> block fails the gate → 0. We send `chat_template_kwargs:
     {enable_thinking: false}` by default and read `message.content` first
     (falling back to reasoning_content ONLY when content is empty, to survive the
     vLLM gemma4 content-empty bug — Fix-A). Serve with thinking disabled.

  3. **score** — verbatim OpenAI grade(): hash-prefix gate then
     difflib.SequenceMatcher ratio in [0,1] (mrcr_helpers.grade). Bench score =
     mean ratio (omk pass_at_1, 0-1 scale). Per-needle means kept for breakdown.

Outputs:
  - <output>.json            — aggregate score + per-needle + provenance
  - <output>.samples.jsonl   — one record per sample (resume-safe append)

CLI matches dispatch_mrcr in omk_eval so template fields → args without translation.

License: dataset is MIT (openai/mrcr). Pulled at runtime, never vendored. Score
artifacts are derivative output, freely publishable per benchmark convention.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # eval/ for cache_sqlite

from mrcr_helpers import grade, iter_bin_samples, BIN_CTX_TOKENS  # noqa: E402

try:
    from cache_sqlite import SqliteResponseCache  # noqa: E402
except Exception:
    SqliteResponseCache = None


class VramSampler:
    """Background peak-VRAM sampler over one physical GPU (nvidia-smi poll).

    omk_eval passes the pinned physical GPU id (plan.gpu_ids[0]) as --vram-gpu so
    the long-context MRCR sweep records the real KV-cache footprint at each bin.
    Pure stdlib + nvidia-smi (always present where llama serves on CUDA); a query
    failure is swallowed so a transient smi hiccup never kills the bench."""

    def __init__(self, gpu: int, period: float = 1.0):
        self.gpu = gpu
        self.period = period
        self.peak_mib = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thr: threading.Thread | None = None

    def _poll_once(self) -> int | None:
        import subprocess
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", "-i", str(self.gpu)],
                timeout=5).decode().strip().splitlines()
            return int(out[0].strip()) if out else None
        except Exception:
            return None

    def _loop(self):
        while not self._stop.is_set():
            v = self._poll_once()
            if v is not None:
                self.peak_mib = max(self.peak_mib, v)
                self.samples += 1
            self._stop.wait(self.period)

    def start(self):
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=3)


def chat_complete(base_url: str, model: str, messages: list[dict],
                  max_tokens: int, timeout: float, enable_thinking: bool) -> dict:
    """Greedy chat completion over the full MRCR message list. Returns
    {text, content, reasoning, prompt_tokens, completion_tokens, finish_reason}.

    `text` is the graded string: content first, reasoning_content only as a
    fallback when content is empty (vLLM gemma4 content-empty bug). Thinking is
    disabled by default via chat_template_kwargs so content leads with the hash."""
    base_payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    payload = dict(base_payload, chat_template_kwargs={"enable_thinking": enable_thinking})
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
    if r.status_code in (400, 422):
        # Some llama-server-family endpoints reject the unknown chat_template_kwargs
        # field. Retry without it — thinking must then be controlled by the served
        # chat template / serve flags (document in the run notes if this path fires).
        r = requests.post(f"{base_url}/v1/chat/completions", json=base_payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    choice = j["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    text = content if content else reasoning
    usage = j.get("usage") or {}
    # llama.cpp / opencoti-llamafile attach a `timings` object to the OAI
    # response: prompt_per_second = PREFILL tok/s, predicted_per_second = GEN
    # tok/s (+ the _n / _ms primitives). vLLM does not emit it — fields stay
    # None and the aggregate just reports null (caller falls back to wall-time).
    tm = j.get("timings") or {}
    return {
        "text": text,
        "content": content,
        "reasoning": reasoning,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "prefill_tok_s": tm.get("prompt_per_second"),
        "gen_tok_s": tm.get("predicted_per_second"),
        "prompt_ms": tm.get("prompt_ms"),
        "predicted_ms": tm.get("predicted_ms"),
        "prompt_n": tm.get("prompt_n"),
        "predicted_n": tm.get("predicted_n"),
    }


def main():
    ap = argparse.ArgumentParser(description="omk-native OpenAI MRCR runner")
    ap.add_argument("--name", required=True,
                    help="Served model name (passed to /v1/chat/completions)")
    ap.add_argument("--base-url", default="http://localhost:8099")
    ap.add_argument("--bin", required=True,
                    help="Context bin name: 256k / 512k / 768k_synth / 1024k / 8k…")
    ap.add_argument("--needles", default="2,4,8",
                    help="Comma list of needle counts to pool (subset of {2,4,8}).")
    ap.add_argument("--num-samples", type=int, default=32,
                    help="Samples per bin (pooled across needle counts).")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="max_tokens for the reproduced answer. MRCR answers are "
                         "full writing pieces (poems/posts); 2048 covers most.")
    ap.add_argument("--http-timeout", type=float, default=1800.0)
    ap.add_argument("--num-concurrent", type=int, default=2,
                    help="ThreadPoolExecutor workers. Keep low at long ctx (KV).")
    ap.add_argument("--enable-thinking", default="false",
                    choices=["true", "false"],
                    help="chat_template_kwargs.enable_thinking. MRCR needs OFF so "
                         "the response begins with the required hash.")
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--vram-gpu", type=int, default=None,
                    help="Physical GPU id to poll for peak VRAM (MiB) during the "
                         "bench. omk_eval passes the pinned plan.gpu_ids[0]. "
                         "Omit to skip VRAM capture.")
    ap.add_argument("--cache-db", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples-cache", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    needles = [int(x) for x in args.needles.split(",") if x.strip()]
    for n in needles:
        if n not in (2, 4, 8):
            print(f"[mrcr] FATAL: needle count {n} not in {{2,4,8}}", flush=True)
            return 11
    enable_thinking = (args.enable_thinking == "true")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path = Path(args.samples_cache) if args.samples_cache else \
        out_path.with_suffix(".samples.jsonl")

    print(f"[mrcr] phase 1: prepare  bin={args.bin} needles={needles} "
          f"n={args.num_samples} ctx_hint={BIN_CTX_TOKENS.get(args.bin)}", flush=True)
    t0 = time.time()
    samples = iter_bin_samples(args.bin, needles, args.num_samples, seed=args.random_seed)
    if not samples:
        print(f"[mrcr] FATAL: no samples landed in bin '{args.bin}' for needles "
              f"{needles} — check the bin band / dataset shards.", flush=True)
        return 12
    print(f"[mrcr] prepared {len(samples)} samples "
          f"(o200k tok range {min(s['o200k_tokens'] for s in samples)}.."
          f"{max(s['o200k_tokens'] for s in samples)})  {time.time()-t0:.0f}s",
          flush=True)

    scache = None
    cache_keys: set[str] = set()
    if args.cache_db and SqliteResponseCache is not None and not args.no_resume:
        scache = SqliteResponseCache(args.cache_db)
        cache_keys = set(scache.keys())

    lock = threading.Lock()
    cache_fp = samples_path.open("a")

    def ckey(s: dict) -> str:
        return f"{args.name}|{args.bin}|{s['sample_id']}|mt{args.max_tokens}|et{enable_thinking}"

    def run_one(s: dict) -> dict:
        key = ckey(s)
        if scache is not None and key in cache_keys:
            rec = dict(scache[key])
            rec["_cached"] = True
            return rec
        t_req = time.time()
        try:
            res = chat_complete(args.base_url, args.name, s["prompt"],
                                args.max_tokens, args.http_timeout, enable_thinking)
            text = res["text"]
        except Exception as e:  # transport/HTTP error — record, score 0
            res = {"text": "", "content": "", "reasoning": "", "error": str(e)[:300]}
            text = ""
        wall_s = time.time() - t_req
        ratio = grade(text, s["answer"], s["random_string"])
        rec = {
            "sample_id": s["sample_id"], "bin": args.bin, "n_needles": s["n_needles"],
            "o200k_tokens": s["o200k_tokens"], "ratio": ratio,
            "prefix_hit": text.startswith(s["random_string"]),
            "resp_chars": len(text), "answer_chars": len(s["answer"]),
            "finish_reason": res.get("finish_reason"),
            "prompt_tokens": res.get("prompt_tokens"),
            "completion_tokens": res.get("completion_tokens"),
            # Server-reported throughput (llama.cpp/opencoti `timings`): prefill =
            # prompt_per_second, gen = predicted_per_second. None on backends that
            # don't emit timings (vLLM). wall_s is the end-to-end request time —
            # kept as a cross-check / fallback denominator.
            "prefill_tok_s": res.get("prefill_tok_s"),
            "gen_tok_s": res.get("gen_tok_s"),
            "prompt_ms": res.get("prompt_ms"),
            "predicted_ms": res.get("predicted_ms"),
            "wall_s": round(wall_s, 3),
            "content_empty": not bool(res.get("content")),
            "error": res.get("error"),
            "response": text[:4000],
        }
        with lock:
            cache_fp.write(json.dumps(rec) + "\n")
            cache_fp.flush()
            if scache is not None:
                scache[key] = rec
        return rec

    vram = None
    if args.vram_gpu is not None:
        vram = VramSampler(args.vram_gpu)
        vram.start()
        print(f"[mrcr] VRAM sampler armed on GPU {args.vram_gpu}", flush=True)

    print(f"[mrcr] phase 2: infer  concurrent={args.num_concurrent} "
          f"think={enable_thinking}  cached={len(cache_keys)}", flush=True)
    recs: list[dict] = []
    done = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, args.num_concurrent)) as pool:
        futs = {pool.submit(run_one, s): s for s in samples}
        for fut in cf.as_completed(futs):
            rec = fut.result()
            recs.append(rec)
            done += 1
            if done % 4 == 0 or done == len(samples):
                cur = statistics.mean([r["ratio"] for r in recs])
                print(f"[mrcr]   {done}/{len(samples)}  running_mean={cur:.4f}",
                      flush=True)
    cache_fp.close()
    if vram is not None:
        vram.stop()

    print("[mrcr] phase 3: score", flush=True)
    ratios = [r["ratio"] for r in recs]
    mean_ratio = statistics.mean(ratios) if ratios else 0.0
    per_needle = {}
    for n in needles:
        sub = [r["ratio"] for r in recs if r["n_needles"] == n]
        per_needle[str(n)] = round(statistics.mean(sub), 6) if sub else None
    empties = sum(1 for r in recs if r["content_empty"])
    prefix_miss = sum(1 for r in recs if not r["prefix_hit"])
    errors = sum(1 for r in recs if r.get("error"))
    elapsed = time.time() - t0

    # Throughput aggregation (server `timings`). Median over the non-error
    # samples that actually carried timings. prefill = prompt eval tok/s (the
    # long-context cost the DCA work targets); gen = decode tok/s. Wall-time
    # median is the end-to-end cross-check. None on backends w/o timings (vLLM).
    def _med(vals):
        vals = [v for v in vals if isinstance(v, (int, float)) and v > 0]
        return round(statistics.median(vals), 2) if vals else None
    ok = [r for r in recs if not r.get("error")]
    prefill_tok_s = _med([r.get("prefill_tok_s") for r in ok])
    gen_tok_s = _med([r.get("gen_tok_s") for r in ok])
    wall_s_median = _med([r.get("wall_s") for r in ok])
    timed_n = sum(1 for r in ok if r.get("prefill_tok_s"))
    vram_peak_mib = vram.peak_mib if (vram is not None and vram.peak_mib) else None

    out = {
        "bench": "mrcr",
        "pass_at_1": round(mean_ratio, 6),        # omk canonical (0-1)
        "accuracy": round(mean_ratio, 6),
        "metric": "sequence_matcher_ratio",
        "bin": args.bin,
        "ctx_tokens": BIN_CTX_TOKENS.get(args.bin),
        "needles": needles,
        "n": len(recs),
        "num_samples": args.num_samples,
        "per_needle_mean": per_needle,
        "o200k_tokens_median": int(statistics.median(
            [r["o200k_tokens"] for r in recs])) if recs else None,
        "prompt_tokens_median": int(statistics.median(
            [r["prompt_tokens"] for r in recs if r.get("prompt_tokens")]))
            if any(r.get("prompt_tokens") for r in recs) else None,
        "content_empty": empties,
        "prefix_miss": prefix_miss,
        "errors": errors,
        "enable_thinking": enable_thinking,
        "max_tokens": args.max_tokens,
        "elapsed_s": round(elapsed, 1),
        # Throughput + footprint (the T87.pD DCA perf overview). prefill/gen are
        # server-reported tok/s medians; vram_peak_mib is the polled peak over
        # the pinned GPU; both null when unavailable (no timings / no --vram-gpu).
        "prefill_tok_s": prefill_tok_s,
        "gen_tok_s": gen_tok_s,
        "wall_s_median": wall_s_median,
        "timed_samples": timed_n,
        "vram_peak_mib": vram_peak_mib,
        "vram_gpu": args.vram_gpu,
        "model": args.name,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n=== MRCR {args.bin}: pass@1={mean_ratio:.4f} "
          f"(n={len(recs)}, per_needle={per_needle}, prefix_miss={prefix_miss}, "
          f"empty={empties}, err={errors})  elapsed={elapsed:.0f}s", flush=True)
    print(f"=== MRCR {args.bin} throughput: prefill={prefill_tok_s} tok/s  "
          f"gen={gen_tok_s} tok/s  wall_median={wall_s_median}s  "
          f"timed={timed_n}/{len(ok)}  vram_peak="
          f"{vram_peak_mib} MiB (gpu {args.vram_gpu})", flush=True)
    if empties > len(recs) * 0.5 and mean_ratio == 0.0:
        print("[mrcr] WARNING: >50% empty content with 0 score — likely thinking "
              "ON or a serving/parser issue (content-empty bug). Check serve flags.",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
