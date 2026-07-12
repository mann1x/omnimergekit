#!/usr/bin/env python3
"""T174.8b — build the Term_Pref PREFERENCE PAIRS by harvesting A2's OWN loops.

Council recipe: Term_Pref negatives = A2's own greedy loop on the same prompt,
detected via n-gram repetition, truncated at the loop-entry boundary. Positives
= the 128e terminating gold we already have in the v2 corpus.

For each loop-prone prompt in the v2 corpus (buckets seeds/constrained/
multilingual/openended), run A2 greedy; KEEP only prompts where A2 detect_loop's;
emit {prompt, gold(128e), loop_neg(A2 truncated), bucket}. Code/retain skipped
(A2 doesn't loop there -> no Term_Pref signal).

Run on bs2 GPU0 (omk python). The 128e teacher is NOT loaded — gold is read
straight from the v2 corpus jsonl.
"""
import argparse
import json
import sys
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
from audit_full_bench import detect_loop  # noqa: E402

A2 = "/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it"
LOOP_BUCKETS = {"seeds", "constrained", "multilingual", "openended"}


def log(m):
    print("[neg %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def loop_onset_truncate(text, max_chars=2000):
    """Find where the degenerate repetition begins and cut shortly after the
    loop ENTERS (so the negative is 'good prefix -> enters loop'). Falls back to
    a hard char cap. Returns the truncated negative."""
    words = text.split()
    if len(words) >= 30:
        # most-common 5-gram in the trailing 50% = the loop body
        tail = words[len(words) // 2:]
        grams = [" ".join(tail[i:i + 5]) for i in range(len(tail) - 5)]
        if grams:
            top, cnt = Counter(grams).most_common(1)[0]
            if cnt >= 3:
                pos = text.find(top)              # FIRST occurrence = loop entry
                if pos > 0:
                    return text[:pos + len(top) * 3][:max_chars]  # entry + ~2 reps
    # line-refrain fallback
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines:
        c = Counter(ln.strip() for ln in lines)
        ref, n = c.most_common(1)[0]
        if n >= 4:
            pos = text.find(ref)
            if pos > 0:
                return text[:pos + len(ref) * 3][:max_chars]
    return text[:max_chars]


def gen_batch(model, tok, prompts, max_new, bs):
    out_txt = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=False)
                 for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               repetition_penalty=1.0, use_cache=True,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(chunk)):
            new = o[j][enc["input_ids"].shape[1]:]
            out_txt.append(tok.decode(new, skip_special_tokens=True).strip())
        log("  gen %d/%d" % (min(i + bs, len(prompts)), len(prompts)))
    return out_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/mnt/sdc/ml/corpora/antiloop_sft_corpus_v2.jsonl")
    ap.add_argument("--out", default="/mnt/sdc/ml/corpora/termpref_pairs.jsonl")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(x) for x in open(args.corpus)]
    pool = [r for r in rows if r.get("bucket") in LOOP_BUCKETS]
    if args.smoke:
        pool = pool[:args.smoke]
    log("loop-prone prompts: %d (of %d corpus rows)" % (len(pool), len(rows)))

    tok = AutoTokenizer.from_pretrained(A2, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        A2, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map={"": 0})
    model.eval()

    a2_out = gen_batch(model, tok, [r["prompt"] for r in pool], args.max_new, args.bs)

    pairs, looped, by_bucket = [], 0, Counter()
    for r, a2 in zip(pool, a2_out):
        if not detect_loop(a2):
            continue                      # only keep prompts where A2 actually loops
        looped += 1
        neg = loop_onset_truncate(a2)
        if len(neg) < 40 or detect_loop(r["completion"]):
            continue                      # neg too short, or gold itself loops
        pairs.append({"prompt": r["prompt"], "gold": r["completion"],
                      "loop_neg": neg, "bucket": r["bucket"]})
        by_bucket[r["bucket"]] += 1

    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log("A2 looped on %d/%d loop-prone prompts; wrote %d pairs %s -> %s" % (
        looped, len(pool), len(pairs), dict(by_bucket), args.out))


if __name__ == "__main__":
    main()
