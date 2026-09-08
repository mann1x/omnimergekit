#!/usr/bin/env python
"""Load + greedy-generate gate for a REAM-built arm.

These arms were written but never loaded. Before spending GPU-hours on an eval we need to
know the checkpoint round-trips: transformers can materialise it (REAM writes experts
UNPACKED as a ModuleList, the base stores them PACKED), every parameter is finite, and the
model emits coherent text under greedy decoding.

A load alone is not the gate -- a model with a broken expert layout still loads and still
reports finite weights; it fails at the first forward. Hence the generate.
"""
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]
TOK = sys.argv[2] if len(sys.argv) > 2 else MODEL

t0 = time.time()
tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
m.eval()
print(f"loaded in {time.time() - t0:.0f}s  params={sum(p.numel() for p in m.parameters())}",
      flush=True)

bad = [n for n, p in m.named_parameters() if not torch.isfinite(p).all()]
print(f"non-finite tensors: {len(bad)}" + (f" -> {bad[:5]}" if bad else " (clean)"), flush=True)

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "What is 17 * 24? Answer with just the number.",
]

ok = not bad
for p in PROMPTS:
    msgs = [{"role": "user", "content": p}]
    # transformers 5.x returns a BatchEncoding here, not a bare tensor -- pass it as kwargs.
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                  return_dict=True, return_tensors="pt").to("cuda:0")
    n_in = enc["input_ids"].shape[1]
    t1 = time.time()
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=128, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    txt = tok.decode(out[0][n_in:], skip_special_tokens=True)
    n_new = out.shape[1] - n_in
    print(f"\n--- prompt: {p}\n[{n_new} tok in {time.time() - t1:.0f}s]\n{txt[:600]}", flush=True)
    # A degenerate arm emits nothing, or one token repeated. Both are load-clean.
    if n_new < 4 or len(set(txt.split())) < 3:
        print(f"DEGENERATE: {n_new} new tokens, {len(set(txt.split()))} unique words", flush=True)
        ok = False

print(f"\n>>> SMOKE_{'OK' if ok else 'FAIL'} {MODEL}", flush=True)
sys.exit(0 if ok else 1)
