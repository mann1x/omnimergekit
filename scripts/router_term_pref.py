#!/usr/bin/env python3
"""T174.8c — ISOLATE Term_Pref router trainer for A2 (62e Gemma-4 26B-A4B MoE).

Council-endorsed minimal experiment to break the ~3% IFEval/constrained loop
floor that output-layer SFT and teacher-Router-KD both failed on. Trains ONLY
the 90 router params (router.proj.weight / router.scale / router.per_expert_scale,
all verified live-trainable by t174_verify_router.py); everything else frozen.

Loss = a_ce * SeqKD_CE(gold)                       # fit 128e terminating gold
     + a_kl * KL(router_S.proj || A2_orig.proj)    # anti-drift SELF-anchor
     + a_tp * Term_Pref                              # reference-free DPO
Term_Pref = -log sigmoid( beta * (meanlogP(gold) - meanlogP(loop_neg)) )  # length-norm
  positive = 128e terminating gold; negative = A2's OWN greedy loop (termpref_pairs).

WHY this differs from the prior Router-KD null: that anchored to the 128e TEACHER
(unreachable for a 62e router -> domain-redistributive drift). Here the KL anchor
is to A2's OWN original router (stay on the 62e manifold), and Term_Pref supplies
the specific loop-exit gradient. Defers expert-LoRA / entropy / IFEval-oversample.

The KL anchor detaches the captured router-input hidden state (GC-safe) and
anchors each layer's proj independently; grad flows to router.proj only there,
while CE+Term_Pref grads reach all router params via the normal forward.
Run on bs2 GPU0, omk python.
"""
import argparse
import json
import math
import random
import time

import torch
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup)

A2 = "/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it"
_RH = {}   # router module name -> detached input hidden state (this fwd)


def log(m):
    print("[tp %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def make_prehook(name):
    def hook(_mod, args):
        _RH[name] = args[0].detach()      # GC-safe: store detached router input
    return hook


def mask_seq(tok, prompt, completion, max_seq):
    """Render chat(user+assistant), label = completion tokens only (-100 on prompt)."""
    ptxt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    ftxt = tok.apply_chat_template([{"role": "user", "content": prompt},
                                    {"role": "assistant", "content": completion}],
                                   add_generation_prompt=False, tokenize=False)
    pid = tok(ptxt, add_special_tokens=False)["input_ids"]
    fid = tok(ftxt, add_special_tokens=False)["input_ids"]
    b = len(pid)
    if fid[:b] != pid:
        b = 0
        for x, y in zip(pid, fid):
            if x != y:
                break
            b += 1
    fid = fid[:max_seq]
    lab = [-100] * min(b, len(fid)) + fid[min(b, len(fid)):]
    return fid, lab


def seq_logprob(logits, ids, labels):
    """MEAN log P(token_t | <t) over completion (label!=-100) tokens.

    Length-normalized (SimPO-style) ON PURPOSE: gold (terminating) and loop_neg
    differ ~4x in length (p50 1293 vs 283 chars in the harvested pairs). Raw SUM
    makes the reference-free DPO margin dominated by length — and since loop_neg
    is the SHORTER side, a sum-based margin would reward LONGER continuations,
    i.e. teach the model NOT to terminate (the opposite of the goal). Per-token
    mean removes that confound: loops carry very high per-token prob (confident
    repetition), so lowering loop_mean / raising gold_mean is the correct
    anti-loop signal.
    """
    lp = logits[:, :-1, :].float().log_softmax(-1)
    tgt = ids[:, 1:]
    lab = labels[:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    m = (lab != -100).float()
    return (tok_lp * m).sum() / m.sum().clamp(min=1.0)


def kl_anchor(router_modules, orig_proj):
    """mean_layers KL( softmax(h@proj_orig^T) || softmax(h@proj_S^T) ), h detached.

    Distillation form: STUDENT in log-space as the kl_div input, detached
    REFERENCE probs as the target. grad w.r.t. input = -ref_probs (finite,
    bounded in [0,1]) — this avoids the log(target) blow-up that made the prior
    form (student as kl_div target) emit a NaN grad on layer-0 router.proj.
    Layer-0 routing is the sparsest softmax (some experts ~0 prob); 0*log0 is
    fine in the VALUE but d/dt[t*log t] = log t + 1 -> -inf as t->0, so the
    target-side grad diverged there. Putting the student on the input side
    removes that singularity. grad -> proj_S only; orig is a detached clone.
    """
    tot, n = 0.0, 0
    for name, mod in router_modules.items():
        h = _RH.get(name)
        if h is None:
            continue
        h2 = h.reshape(-1, h.shape[-1])
        sl = F.linear(h2, mod.proj.weight).float()             # student (grad via proj_S)
        ol = F.linear(h2, orig_proj[name]).float()             # reference (orig_proj detached)
        tot = tot + F.kl_div(sl.log_softmax(-1), ol.softmax(-1).detach(),
                             reduction="batchmean")
        n += 1
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=A2)
    ap.add_argument("--pairs", default="/mnt/sdc/ml/corpora/termpref_pairs.jsonl")
    ap.add_argument("--out", default="/mnt/sdc/ml/google/a2-router-tp")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=2.0)  # SimPO-scale for length-normalized mean-logprob margin (0.1 was sum-scale)
    ap.add_argument("--a-ce", type=float, default=1.0)
    ap.add_argument("--a-kl", type=float, default=0.1)
    ap.add_argument("--a-tp", type=float, default=1.0)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map={"": 0})

    # freeze all, unfreeze router.* only
    router_names = []
    for n, p in model.named_parameters():
        want = ".router." in n
        p.requires_grad_(bool(want))
        if want:
            router_names.append(n)
    router_modules = {n: m for n, m in model.named_modules() if n.endswith(".router")}
    orig_proj = {n: m.proj.weight.detach().clone() for n, m in router_modules.items()}
    hooks = [m.register_forward_pre_hook(make_prehook(n)) for n, m in router_modules.items()]
    log("router params trainable=%d  router modules=%d" % (len(router_names), len(router_modules)))
    assert router_names and len(router_modules) == 30

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()

    rows = [json.loads(x) for x in open(args.pairs)]
    log("pairs=%d" % len(rows))
    data = []
    for r in rows:
        g_ids, g_lab = mask_seq(tok, r["prompt"], r["gold"], args.max_seq)
        l_ids, l_lab = mask_seq(tok, r["prompt"], r["loop_neg"], args.max_seq)
        if sum(1 for x in g_lab if x != -100) < 2 or sum(1 for x in l_lab if x != -100) < 2:
            continue
        data.append((g_ids, g_lab, l_ids, l_lab))
    log("usable pairs=%d" % len(data))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(data) / args.accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup_ratio * steps), steps)
    log("optim steps=%d eff_batch=%d lr=%.1e beta=%.2f a_ce/kl/tp=%.2f/%.2f/%.2f" % (
        steps, args.accum, args.lr, args.beta, args.a_ce, args.a_kl, args.a_tp))

    dev = model.device
    gstep = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        random.shuffle(data)
        opt.zero_grad()
        run = {"ce": 0.0, "kl": 0.0, "tp": 0.0, "acc": 0.0}
        wn = 0
        for i, (g_ids, g_lab, l_ids, l_lab) in enumerate(data):
            gi = torch.tensor([g_ids], device=dev)
            gl = torch.tensor([g_lab], device=dev)
            li = torch.tensor([l_ids], device=dev)
            ll = torch.tensor([l_lab], device=dev)
            _RH.clear()
            go = model(input_ids=gi, attention_mask=torch.ones_like(gi),
                       mm_token_type_ids=torch.zeros_like(gi), use_cache=False)
            ce = F.cross_entropy(go.logits[:, :-1, :].float().reshape(-1, go.logits.size(-1)),
                                 gl[:, 1:].reshape(-1), ignore_index=-100)
            gold_lp = seq_logprob(go.logits, gi, gl)
            kl = kl_anchor(router_modules, orig_proj)
            lo = model(input_ids=li, attention_mask=torch.ones_like(li),
                       mm_token_type_ids=torch.zeros_like(li), use_cache=False)
            loop_lp = seq_logprob(lo.logits, li, ll)
            marg = args.beta * (gold_lp - loop_lp)
            tp = -F.logsigmoid(marg)
            loss = (args.a_ce * ce + args.a_kl * kl + args.a_tp * tp) / args.accum
            loss.backward()
            run["ce"] += ce.item()
            run["kl"] += float(kl)
            run["tp"] += tp.item()
            run["acc"] += float(marg.item() > 0)
            wn += 1
            if (i + 1) % args.accum == 0 or i == len(data) - 1:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad()
                gstep += 1
                if gstep % 5 == 0 or gstep == steps:
                    log("ep%d step %d/%d ce=%.3f kl=%.4f tp=%.3f pref_acc=%.2f lr=%.1e t=%.0fs" % (
                        epoch, gstep, steps, run["ce"]/wn, run["kl"]/wn, run["tp"]/wn,
                        run["acc"]/wn, sched.get_last_lr()[0], time.time()-t0))
                    run = {"ce": 0.0, "kl": 0.0, "tp": 0.0, "acc": 0.0}
                    wn = 0

        # ---- per-epoch FULL-model checkpoint (router edits are in-place on the
        # bf16 weights; save each epoch so Stage-B can pick the gentlest passing
        # epoch per the earliest-passing-epoch anti-regression rule) ----
        ep_out = "%s-ep%d" % (args.out, epoch + 1)
        model.config.use_cache = True
        model.save_pretrained(ep_out, max_shard_size="10GB", safe_serialization=True)
        tok.save_pretrained(ep_out)
        model.config.use_cache = False
        model.train()
        log("SAVED epoch %d -> %s  steps=%d wall=%.0fs" % (epoch + 1, ep_out, gstep, time.time() - t0))

    for h in hooks:
        h.remove()
    log("DONE all epochs  final=%s-ep%d  steps=%d wall=%.0fs" % (args.out, args.epochs, gstep, time.time() - t0))


if __name__ == "__main__":
    main()
