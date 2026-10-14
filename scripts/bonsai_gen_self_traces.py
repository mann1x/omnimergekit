#!/usr/bin/env python
"""Generate the base model's OWN reasoning traces as ternarization calibration text (AYOT, ScaleQ-1.58).

Takes the prompt side (system + first user turn) of chat conversations in a rendered calibration corpus,
lets a llama-server holding the full-precision base continue each one from an open <think>, and writes the
rendered prompt + generation back out. Raw /completion is used so the text is exactly what the model saw
and produced, with no template re-rendering.
"""
import argparse
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--url", default="http://127.0.0.1:8131")
ap.add_argument("--expect-model", required=True, help="substring the served model path must contain")
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--max-prompt-chars", type=int, default=5000)
ap.add_argument("--n-predict", type=int, default=1536)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip", type=int, default=0, help="skip the first N shuffled prompts (held-out split)")
a = ap.parse_args()

props = json.load(urllib.request.urlopen(a.url + "/props", timeout=30))
served = props.get("model_path", "")
if a.expect_model not in served:
    sys.exit(f"FATAL server holds {served!r}, expected {a.expect_model!r}")
print(f"server model: {served}", flush=True)

SYS, ASST = "<|im_start|>system", "<|im_start|>assistant\n"
text = open(a.corpus).read()
prompts, seen = [], set()
for seg in text.split(SYS)[1:]:
    i = seg.find(ASST)
    if i < 0:
        continue
    p = SYS + seg[: i + len(ASST)] + "<think>\n"
    if len(p) > a.max_prompt_chars or p in seen:
        continue
    seen.add(p)
    prompts.append(p)
random.Random(a.seed).shuffle(prompts)
prompts = prompts[a.skip: a.skip + a.n]
print(f"{len(seen)} distinct prompts under {a.max_prompt_chars} chars, using {len(prompts)}", flush=True)


def gen(p):
    body = {"prompt": p, "n_predict": a.n_predict, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "stop": ["<|im_end|>"], "cache_prompt": False}
    req = urllib.request.Request(a.url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return p, r["content"], r.get("tokens_predicted", 0), r.get("stop_type", "")


n_tok = n_done = n_cap = n_empty = 0
with ThreadPoolExecutor(a.workers) as ex, open(a.out, "w") as fo, open(a.out + ".jsonl", "w") as fj:
    for p, c, nt, st in ex.map(gen, prompts):
        n_done += 1
        n_tok += nt
        n_cap += st == "limit"
        if len(c.strip()) < 5:
            n_empty += 1
            continue
        fo.write(p + c + "<|im_end|>\n")
        fj.write(json.dumps({"prompt_chars": len(p), "gen_tokens": nt, "stop": st, "closed_think": "</think>" in c}) + "\n")
        if n_done % 25 == 0:
            print(f"  {n_done}/{len(prompts)}  generated {n_tok} tokens  cap-hit {n_cap}  empty {n_empty}", flush=True)
print(f"TRACES_DONE convs={n_done} gen_tokens={n_tok} cap_hit={n_cap} empty={n_empty}", flush=True)
