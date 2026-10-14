#!/usr/bin/env python
"""Generate the Qwen3-1.7B teacher's OWN reasoning traces for the 1.7B ternary recipe search.

Same prompt extraction as bonsai_gen_self_traces.py (system + first user turn of a rendered chat corpus,
continued from an open <think>), but generated in-process with transformers, since the 1.7B teacher has
no GGUF here. Prompts are shuffled once; the first --heldout go to the eval file, the rest are sampled
--samples times each for the training file, so the two files never share a prompt.
"""
import argparse
import json
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--corpus", required=True, help="rendered chat corpus: prompt source, or (with --jsonl) the corpus to stay disjoint from")
ap.add_argument("--jsonl", default="", help="neutral prompt source: one JSON per line; only --out-eval is written")
ap.add_argument("--field", default="instruction")
ap.add_argument("--empty-field", default="context", help="with --jsonl, keep only rows where this field is empty")
ap.add_argument("--out-train", default="")
ap.add_argument("--out-eval", required=True)
ap.add_argument("--heldout", type=int, default=64)
ap.add_argument("--samples", type=int, default=3)
ap.add_argument("--max-prompt-chars", type=int, default=5000)
ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--bs", type=int, default=96)
ap.add_argument("--limit", type=int, default=0, help="cap the number of training prompts (smoke)")
ap.add_argument("--device", default="cuda:1")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)

SYS, ASST, END = "<|im_start|>system", "<|im_start|>assistant\n", "<|im_end|>"
prompts = set()
if a.jsonl:                                       # user turn only, no system prompt, absent from --corpus
    corpus, n_seen = open(a.corpus).read(), 0
    for line in open(a.jsonl):
        r = json.loads(line)
        q = r[a.field].strip()
        if r.get(a.empty_field, "").strip() or not 20 <= len(q) < a.max_prompt_chars:
            continue
        if q in corpus:
            n_seen += 1
            continue
        prompts.add(f"<|im_start|>user\n{q}<|im_end|>\n{ASST}<think>\n")
    print(f"{n_seen} instructions dropped for appearing in {a.corpus}", flush=True)
else:
    assert a.out_train
    for seg in open(a.corpus).read().split(SYS)[1:]:
        i = seg.find(ASST)
        if 0 <= i < a.max_prompt_chars:
            prompts.add(SYS + seg[: i + len(ASST)] + "<think>\n")
prompts = sorted(prompts)
random.Random(a.seed).shuffle(prompts)
ev, tr = prompts[: a.heldout], ([] if a.jsonl else prompts[a.heldout:])
if a.limit:
    tr = tr[: a.limit]
print(f"{len(prompts)} distinct prompts: {len(ev)} held out, {len(tr)} x {a.samples} for training", flush=True)

tok = AutoTokenizer.from_pretrained(a.base)
tok.padding_side = "left"
end_id = tok.convert_tokens_to_ids(END)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(a.device).eval()


@torch.no_grad()
def run(ps, path):
    ps = sorted(ps, key=len)                      # similar lengths per batch -> little left padding
    n_tok = n_cap = n_empty = n_think = 0
    t0 = time.time()
    with open(path, "w") as fo:
        for b in range(0, len(ps), a.bs):
            chunk = ps[b: b + a.bs]
            enc = tok(chunk, return_tensors="pt", padding=True).to(a.device)
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                                 eos_token_id=end_id, pad_token_id=pad_id)
            for p, row in zip(chunk, out[:, enc.input_ids.shape[1]:].tolist()):
                cap = end_id not in row
                if not cap:
                    row = row[: row.index(end_id)]
                c = tok.decode(row, skip_special_tokens=False)
                if len(c.strip()) < 5:
                    n_empty += 1
                    continue
                n_tok += len(row)
                n_cap += cap
                n_think += "</think>" in c
                fo.write(p + c + END + "\n")
            fo.flush()
            done = min(b + a.bs, len(ps))
            print(f"  {path.rsplit('/', 1)[-1]} {done}/{len(ps)}  {n_tok} tokens  {n_tok / (time.time() - t0):.0f} tok/s", flush=True)
    stats = dict(file=path, convs=len(ps) - n_empty, gen_tokens=n_tok, cap_hit=n_cap, empty=n_empty, closed_think=n_think)
    print("STATS", json.dumps(stats), flush=True)
    return stats


s = [run(ev, a.out_eval)] + ([run(tr * a.samples, a.out_train)] if tr else [])
json.dump(dict(args=vars(a), stats=s), open((a.out_train or a.out_eval) + ".stats.json", "w"), indent=1)
print("TRACES17_DONE", flush=True)
