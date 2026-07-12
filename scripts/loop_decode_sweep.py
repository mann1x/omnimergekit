#!/usr/bin/env python3
"""T182 decode-time anti-repetition sweep on A2 (62e).

Five diagnostics (T180/T176.4/T181) showed the loop is an emergent greedy-decoding
repetition attractor with NO local router signal to gate on. The pragmatic 0%-loops
lever is therefore at the DECODER, not the weights: break the repetition fixed-point
with anti-repetition logit ops. All prior loop numbers (A2 35% multilingual / 15.5%
total) were harvested under the frozen-greedy eval sampler (repetition_penalty=1.0,
the MOST loop-prone config). This is a per-model decode-config sweep, NOT a
cross-variant benchmark, so varying the sampler here does not touch the frozen-greedy
comparison rule.

Decoding stays GREEDY (do_sample=False) so it's deterministic and isolates the
anti-repetition effect; only repetition_penalty / no_repeat_ngram_size vary.
Same 200-prompt loop_screen_sample.jsonl, same detect_loop, same by-bucket, so each
config is directly comparable to A2's published 35%/15.5% baseline. Load A2 once,
sweep configs in-process. Run on bs2 GPU0, omk python.
"""
import argparse
import json
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
from audit_full_bench import detect_loop  # noqa: E402

A2 = "/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it"
# (repetition_penalty, no_repeat_ngram_size); 1.0/0 == frozen-greedy baseline reproduce
CONFIGS = [(1.0, 0), (1.05, 0), (1.1, 0), (1.0, 3)]


def log(m):
    print("[sweep %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=A2)
    ap.add_argument("--sample", default="/mnt/sdc/ml/corpora/loop_screen_sample.jsonl")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--out-dir", default="/srv/ml/eval_results_tracks_2_3/t176_phase3/decode_sweep")
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    rows = [json.loads(x) for x in open(args.sample)]
    prompts = [r["prompt"] for r in rows]
    buckets = [r.get("bucket", "?") for r in rows]
    log("model=%s  sample=%d prompts  configs=%s" % (args.model, len(rows), CONFIGS))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="eager", device_map={"": 0}).eval()

    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     add_generation_prompt=True, tokenize=False) for p in prompts]

    def screen(rp, ng):
        gk = dict(max_new_tokens=args.max_new, do_sample=False, use_cache=True,
                  repetition_penalty=rp, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if ng > 0:
            gk["no_repeat_ngram_size"] = ng
        outs = []
        t0 = time.time()
        for i in range(0, len(texts), args.bs):
            chunk = texts[i:i + args.bs]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            with torch.no_grad():
                o = model.generate(**enc, **gk)
            for j in range(len(chunk)):
                new = o[j][enc["input_ids"].shape[1]:]
                outs.append(tok.decode(new, skip_special_tokens=True).strip())
            if (i // args.bs) % 4 == 0:
                log("    rp=%.2f ng=%d  gen %d/%d (%.0fs)" % (rp, ng, min(i + args.bs, len(texts)),
                                                             len(texts), time.time() - t0))
        looped = [detect_loop(t) for t in outs]
        by = Counter()
        by_loop = Counter()
        for b, lp in zip(buckets, looped):
            by[b] += 1
            if lp:
                by_loop[b] += 1
        n = len(outs)
        nloop = sum(looped)
        return {
            "repetition_penalty": rp, "no_repeat_ngram_size": ng,
            "n": n, "loops": nloop, "loop_pct": round(100.0 * nloop / n, 2),
            "by_bucket": {b: {"n": by[b], "loops": by_loop[b],
                              "pct": round(100.0 * by_loop[b] / by[b], 1)} for b in sorted(by)},
            "wall_s": round(time.time() - t0, 0),
        }

    summary = []
    for rp, ng in CONFIGS:
        log("===== config rep_penalty=%.2f no_repeat_ngram=%d =====" % (rp, ng))
        res = screen(rp, ng)
        tag = "rp%.2f_ng%d" % (rp, ng)
        json.dump(res, open(os.path.join(args.out_dir, "%s.json" % tag), "w"), indent=1)
        ml = res["by_bucket"].get("multilingual", {})
        co = res["by_bucket"].get("constrained", {})
        summary.append(res)
        log("  DONE %s: total=%.2f%% (%d/%d)  multilingual=%.1f%%  constrained=%.1f%%  (%.0fs)" % (
            tag, res["loop_pct"], res["loops"], res["n"],
            ml.get("pct", 0), co.get("pct", 0), res["wall_s"]))

    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1)
    log("=" * 64)
    log("%-16s %8s %14s %12s" % ("config", "total%", "multilingual%", "constrained%"))
    for r in summary:
        ml = r["by_bucket"].get("multilingual", {}).get("pct", 0)
        co = r["by_bucket"].get("constrained", {}).get("pct", 0)
        log("rp%.2f_ng%-7d %7.2f%% %13.1f%% %11.1f%%" % (
            r["repetition_penalty"], r["no_repeat_ngram_size"], r["loop_pct"], ml, co))
    log("DECODE_SWEEP_DONE -> %s/summary.json" % args.out_dir)


if __name__ == "__main__":
    main()
