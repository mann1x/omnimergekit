#!/usr/bin/env python
"""Spot check of teacher self-generation before a full replay-corpus run.

For a stratified sample of prompts (corpus prompts with tools, corpus prompts without tools, neutral instructions)
and several generation arms, ask an OpenAI-compatible server holding the teacher for one completion each and report
how many end on their own, how long they are, how they end (answer / tool call) and why the rest fail.

Arms:
  think         thinking on, system prompt as in the corpus
  think_plain   thinking on, the "Reasoning effort is set to xhigh ..." sentence removed from the system prompt
  nothink       thinking off (<think>\\n\\n</think>\\n\\n prefilled, as the chat template does), system prompt as in the corpus
  think_vendor / nothink_vendor   same prompts as think / nothink with the Qwen3.5 card's general-task samplers
                (thinking: temp 1.0 top_p 0.95 top_k 20 presence 1.5; non-thinking: temp 0.7 top_p 0.8 top_k 20 presence 1.5);
                think / think_plain / nothink use temp 0.6 / 0.6 / 0.7 with NO presence penalty
Every generation is stored in full (kept or not) in <out>.jsonl for reading.
"""
import argparse
import collections
import json
import random
import statistics
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8141")
ap.add_argument("--expect-model", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--jsonl", default="")
ap.add_argument("--jsonl-cat", default="neutral", help="category label of the --jsonl rows")
ap.add_argument("--jsonl-max-chars", type=int, default=2000)
ap.add_argument("--n-tools", type=int, default=60)
ap.add_argument("--n-notools", type=int, default=90)
ap.add_argument("--n-neutral", type=int, default=50)
ap.add_argument("--max-new", type=int, default=8192)
ap.add_argument("--arms", default="think,think_plain,nothink", help="comma list of: " + "think think_plain nothink think_vendor nothink_vendor")
ap.add_argument("--workers", type=int, default=128)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

m = json.load(urllib.request.urlopen(a.url + "/v1/models", timeout=30))["data"][0]
if a.expect_model not in m.get("root", ""):
    sys.exit(f"FATAL server holds {m.get('root')!r}, expected {a.expect_model!r}")
MODEL = m["id"]
SYS, USER, ASST, END = "<|im_start|>system", "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>"
EFFORT = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible "
          "alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\n")

rng = random.Random(a.seed)
tools, plain = [], []
seen = set()
for seg in open(a.corpus).read().split(SYS)[1:]:
    i = seg.find(ASST)
    if not 0 <= i < 5000 or seg[:i] in seen:
        continue
    seen.add(seg[:i])
    (tools if "<tools>" in seg[:i] else plain).append(SYS + seg[:i])
neutral = []
if a.jsonl:
    for line in open(a.jsonl):
        r = json.loads(line)
        if not r.get("context", "").strip() and 20 <= len(r["instruction"]) < a.jsonl_max_chars:
            neutral.append(f"{USER}{r['instruction'].strip()}{END}\n")
sample = [(h, c) for c, pool, n in (("corpus-tools", tools, a.n_tools), ("corpus-notools", plain, a.n_notools), (a.jsonl_cat, neutral, a.n_neutral))
          for h in rng.sample(sorted(pool), min(n, len(pool)))]
n_eff = sum(EFFORT in h for h, _ in sample)
print(f"server {m['root']}; pools tools={len(tools)} notools={len(plain)} neutral={len(neutral)}; sample {len(sample)}, "
      f"{n_eff} carry the xhigh sentence", flush=True)
ARMS = a.arms.split(",")
SAMP = dict(think=dict(temperature=0.6, top_p=0.95, top_k=20), think_plain=dict(temperature=0.6, top_p=0.95, top_k=20),
            nothink=dict(temperature=0.7, top_p=0.8, top_k=20),
            think_vendor=dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5),
            nothink_vendor=dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5))
jobs = [(h, c, arm) for h, c in sample for arm in ARMS if not (arm == "think_plain" and EFFORT not in h)]


def looped(text):                                 # share of non-empty lines in the last 150 that repeat an earlier one
    ls = [x.strip() for x in text.splitlines() if len(x.strip()) > 12][-150:]
    return 1 - len(set(ls)) / len(ls) if len(ls) >= 20 else 0.0


def gen(job):
    head, cat, arm = job
    if arm == "think_plain":
        head = head.replace(EFFORT, "")
    think = not arm.startswith("nothink")
    prompt = head + ASST + ("<think>\n" if think else "<think>\n\n</think>\n\n")
    body = {"model": MODEL, "prompt": prompt, "max_tokens": a.max_new, "skip_special_tokens": False, **SAMP[arm]}
    req = urllib.request.Request(a.url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=7200))
    except Exception as e:                        # noqa: BLE001
        return dict(cat=cat, arm=arm, outcome=f"error:{type(e).__name__}", gen_tokens=0, text="", prompt=prompt)
    ch, text = r["choices"][0], r["choices"][0]["text"]
    for t in (END, "<|endoftext|>"):
        text = text.removesuffix(t)
    k = text.find("</think>")
    body_text = text[k + 8:] if think and k >= 0 else text
    if ch["finish_reason"] != "stop":
        out = "cap_in_thinking" if think and k < 0 else "cap_in_answer"
    elif think and k < 0:
        out = "no_think_close"
    elif "<|im_start|>" in text or text.count("</think>") > (1 if think else 0) or "<think>" in text:
        out = "malformed"
    elif len(body_text.strip()) < 2:
        out = "empty_answer"
    else:
        out = "ok_tool_call" if "<tool_call>" in body_text else "ok_answer"
    return dict(cat=cat, arm=arm, outcome=out, gen_tokens=r["usage"]["completion_tokens"], think_chars=max(k, 0) if think else 0,
                answer_chars=len(body_text.strip()), loop=round(looped(text), 3), prompt=prompt, text=text)


with ThreadPoolExecutor(a.workers) as ex:
    res = list(ex.map(gen, jobs))
with open(a.out + ".jsonl", "w") as fo:
    for r in res:
        fo.write(json.dumps(r) + "\n")

rows = []
for cat in ("corpus-tools", "corpus-notools", a.jsonl_cat):
    for arm in ARMS:
        rs = [r for r in res if r["cat"] == cat and r["arm"] == arm]
        if not rs:
            continue
        ok = [r for r in rs if r["outcome"].startswith("ok")]
        oc = collections.Counter(r["outcome"] for r in rs)
        bad = [r for r in rs if not r["outcome"].startswith("ok")]
        row = dict(cat=cat, arm=arm, n=len(rs), complete=round(len(ok) / len(rs), 3),
                   complete_le2k=round(sum(r["gen_tokens"] <= 2048 for r in ok) / len(rs), 3),
                   complete_le4k=round(sum(r["gen_tokens"] <= 4096 for r in ok) / len(rs), 3),
                   complete_le8k=round(sum(r["gen_tokens"] <= 8192 for r in ok) / len(rs), 3),
                   complete_le16k=round(sum(r["gen_tokens"] <= 16384 for r in ok) / len(rs), 3),
                   median_tokens_ok=statistics.median(r["gen_tokens"] for r in ok) if ok else None,
                   p90_tokens_ok=sorted(r["gen_tokens"] for r in ok)[int(0.9 * len(ok))] if ok else None,
                   failed_looping=sum(r.get("loop", 0) > 0.3 for r in bad), outcomes=dict(oc))
        rows.append(row)
        print("ROW", json.dumps(row), flush=True)
json.dump(dict(args=vars(a), model=m["root"], rows=rows), open(a.out + ".summary.json", "w"), indent=1)
print("SPOTCHECK_DONE", flush=True)
