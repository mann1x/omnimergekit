#!/usr/bin/env python3
"""Can an AtomicChat calib corpus built for tokenizer X be reused to compute an
imatrix for model Y?

A sha256 mismatch between the corpus manifest tokenizer and the target model
tokenizer is NOT the question. The corpus is TEXT; what matters is whether the
target tokenizer maps that text to the SAME id stream -- including the chat
markup, which llama-imatrix only sees as special tokens when given
--parse-special. So measure the id streams, do not argue from the sha.
"""
import json, sys
from transformers import AutoTokenizer

corpus, dir_a, dir_b = sys.argv[1], sys.argv[2], sys.argv[3]
ta = AutoTokenizer.from_pretrained(dir_a)
tb = AutoTokenizer.from_pretrained(dir_b)
text = open(corpus, encoding="utf-8").read()
docs = text.split("\n\n")
print(f"corpus {corpus}: {len(text)} chars, {len(docs)} docs split on the "
      f"manifest separator")

na = nb = 0
mismatched = []
for i, d in enumerate(docs):
    ia = ta(d, add_special_tokens=False)["input_ids"]
    ib = tb(d, add_special_tokens=False)["input_ids"]
    na += len(ia); nb += len(ib)
    if ia != ib:
        mismatched.append(i)
    if i % 500 == 0:
        print(f"  ..{i}/{len(docs)}", flush=True)

print()
print(f"REF  {dir_a}: {na} tokens")
print(f"TGT  {dir_b}: {nb} tokens")
print(f"docs with a DIFFERING id stream: {len(mismatched)} / {len(docs)}")
if mismatched:
    print(f"  first differing doc indices: {mismatched[:10]}")
    i = mismatched[0]
    ia = ta(docs[i], add_special_tokens=False)["input_ids"]
    ib = tb(docs[i], add_special_tokens=False)["input_ids"]
    for j, (x, y) in enumerate(zip(ia, ib)):
        if x != y:
            print(f"  doc {i} pos {j}: ref {x} {ta.decode([x])!r} != "
                  f"tgt {y} {tb.decode([y])!r}")
            break

# The chat markup is what makes --parse-special load-bearing. Prove it is here,
# and that BOTH tokenizers resolve each marker to the same single id.
print()
print("special-token census (occurrences in corpus / id under ref / id under tgt):")
for m in ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>",
          "</tool_call>", "<|endoftext|>"):
    c = text.count(m)
    ra = ta(m, add_special_tokens=False)["input_ids"]
    rb = tb(m, add_special_tokens=False)["input_ids"]
    flag = "" if ra == rb else "   <-- MISMATCH"
    print(f"  {m:16s} {c:8d}  ref={ra} tgt={rb}{flag}")

print()
print("VERDICT: " + ("REUSABLE -- id streams identical"
                     if not mismatched else
                     f"NOT REUSABLE -- {len(mismatched)} docs differ"))
