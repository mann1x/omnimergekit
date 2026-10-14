#!/usr/bin/env python
"""Generate COMPLETE Qwen3-1.7B teacher conversations (thinking + final answer) from a broad prompt pool.

Unlike bonsai17_gen_traces.py (in-process, 1,024-token cap, mostly unfinished reasoning), this drives an
OpenAI-compatible server holding the full-precision teacher with a high token cap and keeps a trace only when
the model ended it itself: finish_reason == "stop", </think> closed, non-empty answer after it. Anything
else is resampled up to --retries times and then dropped (counted, never truncated into the output).

Prompt pool: system + first user turn of a rendered chat corpus, plus instruction rows of a JSONL
(instruction [+ context]). Prompts found in any --exclude-traces file (the held-out eval sets) are removed.
"""
import argparse
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8141")
ap.add_argument("--expect-model", required=True, help="substring the served model root must contain")
ap.add_argument("--corpus", default="")
ap.add_argument("--jsonl", default="")
ap.add_argument("--jsonl-name", default="dolly", help="source label recorded for --jsonl rows")
ap.add_argument("--exclude-traces", nargs="*", default=[])
ap.add_argument("--out", required=True, help="writes <out>.jsonl (text + meta) and <out>.stats.json")
ap.add_argument("--max-prompt-chars", type=int, default=5000)
ap.add_argument("--max-new", type=int, default=8192)
ap.add_argument("--retries", type=int, default=1)
ap.add_argument("--workers", type=int, default=256)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

m = json.load(urllib.request.urlopen(a.url + "/v1/models", timeout=30))["data"][0]
if a.expect_model not in m.get("root", ""):
    sys.exit(f"FATAL server holds {m.get('root')!r}, expected {a.expect_model!r}")
MODEL = m["id"]
print(f"server model: {m['root']} as {MODEL}", flush=True)

SYS, USER, ASST, END = "<|im_start|>system", "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>"
evals = [open(f).read() for f in a.exclude_traces]    # held-out eval sets, written as prompt + generation
pool = {}
if a.corpus:
    for seg in open(a.corpus).read().split(SYS)[1:]:
        i = seg.find(ASST)
        if 0 <= i < a.max_prompt_chars:
            pool.setdefault(SYS + seg[:i], "corpus")
if a.jsonl:
    for line in open(a.jsonl):
        r = json.loads(line)
        q, ctx = r["instruction"].strip(), r.get("context", "").strip()
        if ctx:
            q = f"{q}\n\n{ctx}"
        if 20 <= len(q) < a.max_prompt_chars:
            pool.setdefault(f"{USER}{q}{END}\n", a.jsonl_name + ("-ctx" if ctx else ""))
n_all = len(pool)
pool = {h: s for h, s in pool.items() if not any(h + ASST + "<think>\n" in e for e in evals)}
print(f"{n_all} distinct prompts, {n_all - len(pool)} removed as eval prompts", flush=True)
items = sorted(pool.items())
random.Random(a.seed).shuffle(items)
if a.limit:
    items = items[: a.limit]


def gen(item):
    head, src = item
    prompt = head + ASST + "<think>\n"
    why = "error"
    for attempt in range(a.retries + 1):
        body = {"model": MODEL, "prompt": prompt, "max_tokens": a.max_new, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                "skip_special_tokens": False, "seed": a.seed * 1000003 + hash(head) % 1000003 + attempt}
        req = urllib.request.Request(a.url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=3600))
        except Exception as e:                    # noqa: BLE001  one bad request must not sink the pool
            why = f"error:{type(e).__name__}"
            continue
        ch, nt = r["choices"][0], r["usage"]["completion_tokens"]
        text = ch["text"]
        for t in (END, "<|endoftext|>"):
            text = text.removesuffix(t)
        k = text.find("</think>")
        if ch["finish_reason"] != "stop":
            why = "cap"
        elif k < 0:
            why = "no_think_close"
        elif text.count("<think>") or text.count("</think>") != 1 or "<|im_start|>" in text:
            why = "malformed"
        elif len(text[k + 8:].strip()) < 2:
            why = "empty_answer"
        else:
            return dict(text=prompt + text + END + "\n", src=src, gen_tokens=nt, attempt=attempt, prompt_chars=len(prompt)), "ok"
    return None, why


st, n_tok, t0 = {}, 0, time.time()
with ThreadPoolExecutor(a.workers) as ex, open(a.out + ".jsonl", "w") as fo:
    for k, (rec, why) in enumerate(ex.map(gen, items), 1):
        key = why if rec is None else f"ok:{rec['src']}"
        st[key] = st.get(key, 0) + 1
        if rec:
            n_tok += rec["gen_tokens"]
            fo.write(json.dumps(rec) + "\n")
        if k % 500 == 0 or k == len(items):
            fo.flush()
            print(f"  {k}/{len(items)}  kept tokens {n_tok}  {n_tok / (time.time() - t0):.0f} tok/s  {json.dumps(st, sort_keys=True)}", flush=True)
json.dump(dict(args=vars(a), prompts=len(items), kept_tokens=n_tok, outcome=st), open(a.out + ".stats.json", "w"), indent=1)
print("TRACES17_SRV_DONE", json.dumps(st, sort_keys=True), flush=True)
