#!/usr/bin/env python3
"""Stage 1 of the CoderX brevity distillation (#845): sample k solutions per
held-out LCB problem from ONE served model and verify each by execution.

This is the generation half. It produces, for every (problem, sample) pair, the
full completion split into its reasoning and answer channels plus a hard
pass/fail from the LCB scorer. Stage 2 (`brevity_pairs.py`) joins two of these
files into preference pairs.

WHY THE CHANNELS ARE KEPT SEPARATE
----------------------------------
`lcb_llama_server.chat_complete` concatenates reasoning + content into one
`text`, which is all the eval path needs. Training does NOT: the thing we are
trying to shorten is the THINKING channel ("it thinks too much"), and a trainer
has to re-render the two channels into the model's own chat format. Collapsing
them here would make the split unrecoverable, so this module issues its own
request and stores `reasoning` and `content` as separate fields. `text` is kept
too, identical to what the eval path would have produced, so a row can be
scored by exactly the same code the banked cells used.

BASIS
-----
Every knob defaults to the banked qwen_suite/lcb_v6_77q cell:
    sampler `recommended` (t 0.6 / top_p 0.95 / top_k 20)
    max_gen_toks 32768 · thinking_token_budget 12288
    server: -c 262144 --parallel 8 --reasoning-format deepseek
            --reasoning-budget 12288
A pair is only meaningful if BOTH sides were drawn on the same basis, so the
resolved sampler is written into every row and Stage 2 refuses to pair rows
whose basis differs. [[feedback_sampler_is_a_cohort_fact_read_it]]

HOLDOUT
-------
The pool (`eval/lcb/lcb_rl_pool.jsonl`) was built by `build_lcb_rl_pool.py`,
which asserts 0 overlap with every frozen `*taskids.json` eval list. Do not
point this at an eval task-id list — that trains on the test set.

Two phases, both resumable, because the GPU half is expensive and the CPU half
is not:
  A. generate  — threaded HTTP, writes rows with passed=null
  B. verify    — single-threaded, fills passed/why for any row still null
Re-running skips (task_id, sample_idx) rows that already exist AND are verified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from eval.lcb.lcb_helpers import (  # noqa: E402
    clean_lcb_completion,
    score_lcb_problem,
)


def chat(base_url: str, model: str, prompt: str, *, max_tokens: int,
         sampler: dict, thinking_budget: int | None,
         timeout: float) -> dict:
    """One /v1/chat/completions call, reasoning and content kept APART.

    Payload contract mirrors eval/lcb/lcb_llama_server.py::chat_complete —
    sampler keys at the JSON root, thinking_token_budget at the root, and
    enable_thinking inside chat_template_kwargs. Keep the two in step if that
    contract ever changes.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    for k in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty"):
        if sampler.get(k) is not None:
            payload[k] = sampler[k]
    if thinking_budget:
        payload["thinking_token_budget"] = int(thinking_budget)
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload,
                      timeout=timeout)
    r.raise_for_status()
    j = r.json()
    ch = j["choices"][0]
    msg = ch["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    usage = j.get("usage") or {}
    return {
        "content": content,
        "reasoning": reasoning,
        # identical to what chat_complete would have returned, so the scorer
        # sees byte-for-byte what the banked eval cells fed it
        "text": (reasoning + "\n" + content) if (content and reasoning)
                else (content or reasoning),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": ch.get("finish_reason"),
    }


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    with p.open() as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=str(REPO / "eval/lcb/lcb_rl_pool.jsonl"))
    ap.add_argument("--base-url", default="http://localhost:8471")
    ap.add_argument("--model", required=True, help="served-model-name")
    ap.add_argument("--role", required=True, choices=["teacher", "student"],
                    help="teacher = the SHORT model (coder+MPE); "
                         "student = the model being trained (CoderX)")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("--k", type=int, default=4, help="samples per problem")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole pool")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--thinking-budget", type=int, default=12288)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--sampler-name", default="recommended",
                    help="recorded in every row; Stage 2 refuses to pair "
                         "rows whose (sampler, max_tokens) differ")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent requests; match llama-server --parallel")
    ap.add_argument("--http-timeout", type=float, default=1800.0)
    ap.add_argument("--verify-timeout", type=float, default=15.0)
    ap.add_argument("--skip-verify", action="store_true",
                    help="phase A only; run again without it to verify")
    args = ap.parse_args()

    pool = load_jsonl(Path(args.pool))
    if args.limit:
        pool = pool[:args.limit]
    if not pool:
        print(f"REFUSE: empty pool at {args.pool}")
        return 2
    probs = {p["id"]: p for p in pool}
    print(f"pool: {len(pool)} problems x k={args.k} = {len(pool) * args.k} samples")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(out_p)
    have = {(r["task_id"], r["sample_idx"]) for r in existing}
    print(f"resume: {len(existing)} rows already on disk")

    sampler = {"temperature": args.temperature, "top_p": args.top_p,
               "top_k": args.top_k}
    basis = {"sampler_name": args.sampler_name, "sampler": sampler,
             "max_tokens": args.max_tokens,
             "thinking_budget": args.thinking_budget}

    # ---------------------------------------------------------------- phase A
    todo = [(p, i) for p in pool for i in range(args.k)
            if (p["id"], i) not in have]
    print(f"phase A: {len(todo)} samples to generate")
    lock = threading.Lock()
    fh = out_p.open("a")
    done = {"n": 0, "err": 0}
    t0 = time.time()

    def work(item):
        prob, idx = item
        try:
            g = chat(args.base_url, args.model, prob["prompt"],
                     max_tokens=args.max_tokens, sampler=sampler,
                     thinking_budget=args.thinking_budget,
                     timeout=args.http_timeout)
        except Exception as e:  # noqa: BLE001 - one bad sample must not kill the run
            with lock:
                done["err"] += 1
                print(f"  ERR {prob['id']}#{idx}: {type(e).__name__}: {e}")
            return
        row = {
            "task_id": prob["id"], "sample_idx": idx, "role": args.role,
            "model": args.model,
            "content": g["content"], "reasoning": g["reasoning"],
            "completion_tokens": g["completion_tokens"],
            "prompt_tokens": g["prompt_tokens"],
            "finish_reason": g["finish_reason"],
            "passed": None, "why": None,
            "basis": basis,
        }
        with lock:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            done["n"] += 1
            n = done["n"]
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  gen {n}/{len(todo)}  {el / 60:.1f}m  "
                      f"({n / max(el, 1) * 60:.1f}/min)  err={done['err']}")

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))
    fh.close()
    print(f"phase A done: +{done['n']} rows, {done['err']} errors")

    if args.skip_verify:
        return 0

    # ---------------------------------------------------------------- phase B
    # Sequential and in the MAIN process on purpose: score_lcb_problem forks a
    # sandbox child per problem, and forking from a worker thread of a pool that
    # is still alive is how bug-622 (daemonic child) bit us. Verification is
    # cheap next to generation; do not "optimise" this into the thread pool.
    rows = load_jsonl(out_p)
    unver = [r for r in rows if r.get("passed") is None]
    print(f"phase B: {len(unver)}/{len(rows)} rows need verification")
    for i, r in enumerate(unver, 1):
        prob = probs.get(r["task_id"])
        if prob is None:
            r["passed"], r["why"] = False, "task_id not in pool"
            continue
        meta = prob.get("meta") or {}
        text = (r["reasoning"] + "\n" + r["content"]) if (r["reasoning"] and r["content"]) \
            else (r["content"] or r["reasoning"])
        code = clean_lcb_completion(text, meta.get("starter_code") or "")
        try:
            ok, why = score_lcb_problem(code, meta.get("tests") or [],
                                        meta.get("method_name"),
                                        timeout=args.verify_timeout)
        except Exception as e:  # noqa: BLE001
            ok, why = False, f"{type(e).__name__}: {e}"
        r["passed"], r["why"] = bool(ok), str(why)[:200]
        if i % 50 == 0 or i == len(unver):
            print(f"  verify {i}/{len(unver)}")

    tmp = out_p.with_suffix(out_p.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, out_p)

    n_pass = sum(1 for r in rows if r.get("passed"))
    caps = sum(1 for r in rows if r.get("finish_reason") == "length")
    solved = len({r["task_id"] for r in rows if r.get("passed")})
    print(f"\n=== {args.role} / {args.model} ===")
    print(f"  samples      : {len(rows)}")
    print(f"  passed       : {n_pass} ({100 * n_pass / max(len(rows), 1):.1f}%)")
    print(f"  problems >=1 : {solved}/{len(pool)} "
          f"({100 * solved / max(len(pool), 1):.1f}%)")
    print(f"  cap-hit      : {caps} ({100 * caps / max(len(rows), 1):.1f}%)")
    print(f"BREVITY_GEN_OK {args.role} {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
