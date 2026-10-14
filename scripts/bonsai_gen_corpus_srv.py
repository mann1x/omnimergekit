#!/usr/bin/env python
"""Generate a teacher replay corpus of COMPLETE conversations from a plan (see bonsai_corpus_plan.py).

Each plan row names its mode and sampler. thinking on: the prompt ends in "<think>\\n" and a trace is kept only if the
model stopped by itself, closed </think> exactly once and wrote an answer (or a tool call) after it. thinking off: the
prompt ends in the template's empty think block and a trace is kept only if it stopped by itself, contains no think tag
and is not empty. Anything else is resampled up to --retries times, then dropped and counted; nothing truncated is kept.
Resumable: ids already in <out>.jsonl or <out>.dropped.jsonl are skipped.

Samplers (Qwen3.5 model card): think_vendor t1.0 top_p0.95 top_k20 presence1.5; think_code t0.6 top_p0.95 top_k20
presence0; nothink_vendor t0.7 top_p0.8 top_k20 presence1.5; nothink_code t0.7 top_p0.8 top_k20 presence0.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8142")
ap.add_argument("--expect-model", required=True, help="substring the served model root must contain")
ap.add_argument("--plan", required=True)
ap.add_argument("--out", required=True, help="writes <out>.jsonl, <out>.dropped.jsonl and <out>.stats.json")
ap.add_argument("--max-new-think", type=int, default=32768)
ap.add_argument("--max-new-nothink", type=int, default=16384)
ap.add_argument("--retries", type=int, default=1)
ap.add_argument("--workers", type=int, default=192)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

m = json.load(urllib.request.urlopen(a.url + "/v1/models", timeout=30))["data"][0]
if a.expect_model not in m.get("root", ""):
    sys.exit(f"FATAL server holds {m.get('root')!r}, expected {a.expect_model!r}")
MODEL = m["id"]
print(f"server model: {m['root']} as {MODEL}", flush=True)
ASST, END = "<|im_start|>assistant\n", "<|im_end|>"
SAMP = dict(think_vendor=dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5),
            think_code=dict(temperature=0.6, top_p=0.95, top_k=20),
            nothink_vendor=dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5),
            nothink_code=dict(temperature=0.7, top_p=0.8, top_k=20))
plan = [json.loads(x) for x in open(a.plan)]
done = set()
for f in (a.out + ".jsonl", a.out + ".dropped.jsonl"):
    if os.path.exists(f):
        done |= {json.loads(x)["id"] for x in open(f)}
items = [r for r in plan if r["id"] not in done]
if a.limit:
    items = items[: a.limit]
print(f"plan {len(plan)}, already done {len(done)}, to generate {len(items)}", flush=True)


def gen(r):
    think = r["mode"] == "think"
    prompt = r["head"] + ASST + ("<think>\n" if think else "<think>\n\n</think>\n\n")
    why, nt = "error", 0
    for attempt in range(a.retries + 1):
        body = {"model": MODEL, "prompt": prompt, "max_tokens": a.max_new_think if think else a.max_new_nothink, "skip_special_tokens": False,
                "seed": (a.seed * 1000003 + zlib.crc32(r["id"].encode()) + attempt) % 2**31, **SAMP[r["sampler"]]}
        req = urllib.request.Request(a.url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            o = json.load(urllib.request.urlopen(req, timeout=14400))
        except Exception as e:                    # noqa: BLE001  one bad request must not sink the pool
            why = f"error:{type(e).__name__}"
            continue
        ch, nt = o["choices"][0], o["usage"]["completion_tokens"]
        text = ch["text"]
        for t in (END, "<|endoftext|>"):
            text = text.removesuffix(t)
        k = text.find("</think>")
        answer = text[k + 8:] if think and k >= 0 else text
        if ch["finish_reason"] != "stop":
            why = "cap"
        elif think and k < 0:
            why = "no_think_close"
        elif "<think>" in text or text.count("</think>") != (1 if think else 0) or "<|im_start|>" in text or END in text:
            why = "malformed"
        elif len(answer.strip()) < 2:
            why = "empty_answer"
        else:
            return dict(id=r["id"], fam=r["fam"], src=r["src"], mode=r["mode"], sampler=r["sampler"], gen_tokens=nt, attempt=attempt,
                        think_chars=max(k, 0) if think else 0, tool_call="<tool_call>" in answer, text=prompt + text + END + "\n"), "ok"
    return dict(id=r["id"], fam=r["fam"], mode=r["mode"], why=why, gen_tokens=nt), why


st, n_tok, t0 = {}, 0, time.time()
with ThreadPoolExecutor(a.workers) as ex, open(a.out + ".jsonl", "a") as fo, open(a.out + ".dropped.jsonl", "a") as fd:
    for k, (rec, why) in enumerate(ex.map(gen, items), 1):
        key = f"{rec['fam']}:{rec['mode']}:{why}"
        st[key] = st.get(key, 0) + 1
        if why == "ok":
            n_tok += rec["gen_tokens"]
            fo.write(json.dumps(rec) + "\n")
        else:
            fd.write(json.dumps(rec) + "\n")
        if k % 250 == 0 or k == len(items):
            fo.flush()
            fd.flush()
            print(f"  {k}/{len(items)}  kept tokens {n_tok}  {n_tok / (time.time() - t0):.0f} tok/s  {json.dumps(st, sort_keys=True)}", flush=True)
json.dump(dict(args=vars(a), model=m["root"], generated_now=len(items), kept_tokens_now=n_tok, outcome_now=st), open(a.out + ".stats.json", "w"), indent=1)
print("CORPUS_GEN_DONE", json.dumps(st, sort_keys=True), flush=True)
