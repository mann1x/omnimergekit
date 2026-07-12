#!/usr/bin/env python3
"""T174.3 — gentle anti-loop LoRA SFT on A2 (62e Gemma-4 26B-A4B MoE).

Hand-rolled causal-LM SFT (no trl) that adapts router_kd.py's load/save shape but
trains LoRA adapters on the language-model attention + shared dense MLP (the
repetition/termination path), distilling the 128e teacher's terminating
trajectories from the anti-loop corpus. Loss is masked to completion tokens only.

Validated by t174_preflight.py: the regex scopes LoRA to 410 LM adapters (no
vision/experts/router); forward with mm_token_type_ids gives finite loss; grads
flow. Saves ONLY the adapter (no 100 GB copytree); merge happens at build time.

Run on bs2 GPU0 with the omnimergekit python.
"""
import argparse
import json
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup)

A2 = "/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it"
LORA_REGEX = (r"model\.language_model\.layers\.\d+\."
              r"(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)")


def log(m):
    print("[sft %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def build_examples(tok, rows, max_seq):
    """Render each {prompt,completion} with the chat template; mask labels to the
    completion. Returns list of (input_ids, labels)."""
    out = []
    skipped = 0
    for r in rows:
        p, c = r["prompt"], r["completion"]
        ptxt = tok.apply_chat_template([{"role": "user", "content": p}],
                                       add_generation_prompt=True, tokenize=False)
        ftxt = tok.apply_chat_template([{"role": "user", "content": p},
                                        {"role": "assistant", "content": c}],
                                       add_generation_prompt=False, tokenize=False)
        pid = tok(ptxt, add_special_tokens=False)["input_ids"]
        fid = tok(ftxt, add_special_tokens=False)["input_ids"]
        # boundary: prompt prefix should align; fall back to min-length guard
        b = len(pid)
        if fid[:b] != pid:
            # tokenization drift at the seam; locate by longest common prefix
            b = 0
            for x, y in zip(pid, fid):
                if x != y:
                    break
                b += 1
        if len(fid) <= b + 1:
            skipped += 1
            continue
        fid = fid[:max_seq]
        labels = [-100] * min(b, len(fid)) + fid[min(b, len(fid)):]
        out.append((fid, labels))
    log("examples=%d skipped=%d" % (len(out), skipped))
    return out


def collate(batch, pad_id):
    m = max(len(x[0]) for x in batch)
    ids, lab, att = [], [], []
    for fid, labels in batch:
        pad = m - len(fid)
        ids.append(fid + [pad_id] * pad)
        lab.append(labels + [-100] * pad)
        att.append([1] * len(fid) + [0] * pad)
    return (torch.tensor(ids), torch.tensor(lab), torch.tensor(att))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=A2)
    ap.add_argument("--corpus", default="/mnt/sdc/ml/corpora/antiloop_sft_corpus.jsonl")
    ap.add_argument("--out", default="/mnt/sdc/ml/google/a2-antiloop-lora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    log("load student bf16 -> GPU0: %s" % args.base)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map={"": 0})
    cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                     bias="none", task_type="CAUSAL_LM", target_modules=LORA_REGEX)
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and ("lora_" in n) and
           (".experts." in n or ".router." in n or "vision" in n)]
    assert not bad, "LoRA leaked into experts/router/vision: %s" % bad[:3]
    model.config.use_cache = False
    # full-vocab logits (V=262k) over 2048-token batches OOM a 14B MoE without
    # activation checkpointing; enable it + let grads reach adapters through the
    # frozen, checkpointed base.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    rows = [json.loads(x) for x in open(args.corpus)]
    log("corpus rows=%d  buckets=%s" % (
        len(rows), {b: sum(1 for r in rows if r.get("bucket") == b)
                    for b in {r.get("bucket") for r in rows}}))
    data = build_examples(tok, rows, args.max_seq)
    pad_id = tok.pad_token_id

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(data) / (args.bs * args.accum))
    total = steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup_ratio * total), total)
    log("optim steps total=%d (per-epoch=%d) eff_batch=%d" % (
        total, steps_per_epoch, args.bs * args.accum))

    gstep = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        random.shuffle(data)
        micro = 0
        opt.zero_grad()
        run, wn = 0.0, 0
        for i in range(0, len(data), args.bs):
            batch = data[i:i + args.bs]
            ids, lab, att = collate(batch, pad_id)
            ids, lab, att = ids.to(model.device), lab.to(model.device), att.to(model.device)
            out = model(input_ids=ids, attention_mask=att,
                        mm_token_type_ids=torch.zeros_like(ids), use_cache=False)
            logits = out.logits[:, :-1, :].float()
            tgt = lab[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
            (loss / args.accum).backward()
            run += loss.item()
            wn += 1
            micro += 1
            if micro % args.accum == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad()
                gstep += 1
                if gstep % 10 == 0 or gstep == total:
                    log("epoch %d step %d/%d loss=%.4f lr=%.2e elapsed=%.0fs" % (
                        epoch, gstep, total, run / max(wn, 1), sched.get_last_lr()[0],
                        time.time() - t0))
                    run, wn = 0.0, 0
        ep_dir = "%s/epoch%d" % (args.out, epoch + 1)
        model.save_pretrained(ep_dir)
        tok.save_pretrained(ep_dir)
        log("SAVED adapter -> %s" % ep_dir)
    log("DONE total_steps=%d wall=%.0fs" % (gstep, time.time() - t0))


if __name__ == "__main__":
    main()
