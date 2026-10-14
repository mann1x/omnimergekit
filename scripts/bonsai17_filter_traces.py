#!/usr/bin/env python
"""Drop from a trace JSONL every record whose prompt is the prompt of a held-out eval trace. Prints the counts."""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--exclude-traces", nargs="+", required=True)
a = ap.parse_args()
evals = [open(f).read() for f in a.exclude_traces]
MARK = "<|im_start|>assistant\n<think>\n"
kept = dropped = 0
src = {}
with open(a.out, "w") as fo:
    for line in open(a.inp):
        r = json.loads(line)
        head = r["text"][: r["text"].index(MARK) + len(MARK)]
        if any(head in e for e in evals):
            dropped += 1
            continue
        kept += 1
        src[r["src"]] = src.get(r["src"], 0) + 1
        fo.write(line)
print(json.dumps(dict(kept=kept, dropped_eval_prompts=dropped, by_source=src)))
