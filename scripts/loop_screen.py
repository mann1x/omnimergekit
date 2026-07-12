#!/usr/bin/env python3
"""T175 — fast bf16 degenerate-loop SCREEN for the 66e/68e/72e expert-count sweep.

Council (csl-2026-05-31-1658-3956) ruled A2's ~3% IFEval/constrained loop floor a
STRUCTURAL Pareto limit of the 62e prune (8th-vs-9th expert routing margin is
knife-edge), with the falsifiable prediction: a higher-count prune (same
fc15_25-p8 recipe) shows the loop floor collapse with NO post-hoc tuning. This
screen measures that directly and cheaply: run a fixed loop-prone prompt sample
greedy through each pristine bf16 variant and count detect_loop() hits (the SAME
high-precision gate audit_full_bench.py uses for the published 3% number).

bf16 HF generate is used on purpose: it is exactly how A2's own loops were
harvested (gen_loop_negatives.py), so the measurement is consistent with the 3%
anchor's failure population, and it skips the convert+quant per variant — Q6_K is
near-lossless and loop behaviour is invariant to 6-bit quant, so the GGUF is only
built for the winning N in the confirm stage.

Greedy (do_sample=False, repetition_penalty=1.0) — matching the canonical FROZEN
greedy eval sampler; any temperature>0 would mask loops behind sampling noise.

Run on bs2, one model per Blackwell (CUDA_VISIBLE_DEVICES-pinned), omk python.
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


def log(m):
    print("[screen %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sample", default="/mnt/sdc/ml/corpora/loop_screen_sample.jsonl")
    ap.add_argument("--out", required=True, help="per-model result json")
    ap.add_argument("--name", default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=2048)
    args = ap.parse_args()
    name = args.name or args.model.rstrip("/").split("/")[-1]

    rows = [json.loads(x) for x in open(args.sample)]
    log("model=%s  sample=%d prompts" % (name, len(rows)))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="eager", device_map={"": 0})
    model.eval()

    prompts = [r["prompt"] for r in rows]
    buckets = [r.get("bucket", "?") for r in rows]
    outs = []
    t0 = time.time()
    for i in range(0, len(prompts), args.bs):
        chunk = prompts[i:i + args.bs]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=False)
                 for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                               repetition_penalty=1.0, use_cache=True,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(chunk)):
            new = o[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True).strip())
        log("  gen %d/%d  (%.0fs)" % (min(i + args.bs, len(prompts)), len(prompts), time.time() - t0))

    looped = [detect_loop(t) for t in outs]
    by_bucket = Counter()
    by_bucket_loop = Counter()
    examples = []
    for b, lp, txt, pr in zip(buckets, looped, outs, prompts):
        by_bucket[b] += 1
        if lp:
            by_bucket_loop[b] += 1
            if len(examples) < 6:
                examples.append({"bucket": b, "prompt": pr[:160], "tail": txt[-200:]})
    n = len(outs)
    nloop = sum(looped)
    res = {
        "name": name, "model": args.model, "n": n, "loops": nloop,
        "loop_pct": round(100.0 * nloop / max(n, 1), 2),
        "by_bucket": {b: {"n": by_bucket[b], "loops": by_bucket_loop[b]} for b in sorted(by_bucket)},
        "examples": examples,
        "wall_s": round(time.time() - t0, 0),
    }
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    log("DONE %s  loops=%d/%d (%.2f%%)  by_bucket=%s  -> %s" % (
        name, nloop, n, res["loop_pct"],
        {b: "%d/%d" % (by_bucket_loop[b], by_bucket[b]) for b in sorted(by_bucket)}, args.out))


if __name__ == "__main__":
    main()
