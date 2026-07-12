#!/usr/bin/env python3
"""T188 anchor audit: is 128e=0%% a FAIR anchor for the loop floor?

For the multilingual bucket (native-language questions — Turkish/Hindi/etc.),
run 128e and A2 through the EXACT loop_screen path (greedy bf16, same chat
template, same detect_loop) and, for every prompt A2 loops on, print what 128e
actually produced. If 128e answers the question in-language (real capability),
A2's loop is a genuine prune-induced capability loss and 128e=0%% is a fair
anchor. If 128e dodges into English/generic non-looping filler, the anchor is
soft and "loop%%" is partly measuring failure-MODE, not failure-RATE.

GPU1-pinned (CUDA_VISIBLE_DEVICES=1), sequential load (52 GB each).
"""
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
from audit_full_bench import detect_loop  # noqa: E402

SAMPLE = "/mnt/sdc/ml/corpora/loop_screen_sample.jsonl"
M128 = "/srv/ml/models/base/gemma-4-26B-A4B-it"
A2 = "/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it"
OUT = "/srv/ml/eval_results_tracks_2_3/t188_anchor_audit.json"
N = 15


def log(m):
    print("[anchor %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def gen(model_dir, prompts, max_new=2048, bs=8):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation="eager", device_map={"": 0})
    model.eval()
    outs = []
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
            outs.append(tok.decode(new, skip_special_tokens=True).strip())
        log("  %s gen %d/%d" % (model_dir.split("/")[-1], min(i + bs, len(prompts)), len(prompts)))
    del model
    torch.cuda.empty_cache()
    return outs


def main():
    rows = [json.loads(x) for x in open(SAMPLE)]
    ml = [r["prompt"] for r in rows if r.get("bucket") == "multilingual"][:N]
    log("multilingual prompts: %d" % len(ml))
    log("=== generating 128e ===")
    o128 = gen(M128, ml)
    log("=== generating A2 ===")
    oa2 = gen(A2, ml)

    res = []
    for p, c, a in zip(ml, o128, oa2):
        res.append({"prompt": p,
                    "128e_loop": bool(detect_loop(c)), "128e_len": len(c), "128e_out": c,
                    "a2_loop": bool(detect_loop(a)), "a2_len": len(a), "a2_out": a})
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)

    n128 = sum(r["128e_loop"] for r in res)
    na2 = sum(r["a2_loop"] for r in res)
    print("\n========== ANCHOR AUDIT SUMMARY ==========")
    print("multilingual subset n=%d   128e loops=%d   A2 loops=%d" % (len(res), n128, na2))
    print("\n-- per-prompt (loop? / out-len) --")
    for r in res:
        print("  128e[%s %5d]  A2[%s %5d]  %s" % (
            "L" if r["128e_loop"] else ".", r["128e_len"],
            "L" if r["a2_loop"] else ".", r["a2_len"], r["prompt"][:70]))
    print("\n-- WHERE A2 LOOPS: what did 128e produce? (the anchor-fairness test) --")
    any_a2 = False
    for r in res:
        if r["a2_loop"]:
            any_a2 = True
            print("\nPROMPT:", r["prompt"][:140])
            print("  128e (loop=%s, %d ch): %s" % (r["128e_loop"], r["128e_len"], r["128e_out"][:280].replace("\n", " ")))
            print("  A2   (loop=%s, %d ch) tail: %s" % (r["a2_loop"], r["a2_len"], r["a2_out"][-160:].replace("\n", " ")))
    if not any_a2:
        print("  (A2 looped on none of the first %d multilingual prompts — widen N)" % N)
    print("\n[anchor] full outputs -> %s" % OUT)


if __name__ == "__main__":
    main()
