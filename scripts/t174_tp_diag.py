#!/usr/bin/env python3
"""T174.8 NaN diagnostic for router_term_pref.py.

Training went ce=nan from step 5. The base bf16/eager forward is finite
(the harvest job generated finite output on these same long sequences), so
the NaN is in the TRAIN path: gradient-checkpointing, a loss term, or backward.
This isolates the stage by running the exact per-pair computation on pair[0]
with isfinite checks after every stage, NO optimizer step. Run bs2 GPU0, omk py.
"""
import json

import torch
import torch.nn.functional as F

from router_term_pref import (A2, kl_anchor, make_prehook, mask_seq,
                              seq_logprob)
from transformers import AutoModelForCausalLM, AutoTokenizer

PAIRS = "/mnt/sdc/ml/corpora/termpref_pairs.jsonl"


def fin(t):
    if isinstance(t, torch.Tensor):
        return bool(torch.isfinite(t).all())
    return bool(t == t and t not in (float("inf"), float("-inf")))


def main():
    tok = AutoTokenizer.from_pretrained(A2, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        A2, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map={"": 0})

    for n, p in model.named_parameters():
        p.requires_grad_(".router." in n)
    router_modules = {n: m for n, m in model.named_modules() if n.endswith(".router")}
    orig_proj = {n: m.proj.weight.detach().clone() for n, m in router_modules.items()}

    r = json.loads(open(PAIRS).readline())
    g_ids, g_lab = mask_seq(tok, r["prompt"], r["gold"], 2048)
    l_ids, l_lab = mask_seq(tok, r["prompt"], r["loop_neg"], 2048)
    dev = model.device
    gi = torch.tensor([g_ids], device=dev)
    gl = torch.tensor([g_lab], device=dev)
    li = torch.tensor([l_ids], device=dev)
    ll = torch.tensor([l_lab], device=dev)
    print("pair0 gold_tok=%d (lbl=%d) loop_tok=%d (lbl=%d)" % (
        gi.numel(), int((gl != -100).sum()), li.numel(), int((ll != -100).sum())))

    # ---- A: EVAL, no grad-checkpoint (matches inference) ----
    model.eval()
    with torch.no_grad():
        go = model(input_ids=gi, attention_mask=torch.ones_like(gi),
                   mm_token_type_ids=torch.zeros_like(gi), use_cache=False)
    print("A eval-fwd  logits_finite=%s  max|logit|=%.1f" % (
        fin(go.logits), go.logits.float().abs().max().item()))

    # ---- B: TRAIN + grad-checkpoint (the real path) ----
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    hooks = [m.register_forward_pre_hook(make_prehook(n)) for n, m in router_modules.items()]

    go = model(input_ids=gi, attention_mask=torch.ones_like(gi),
               mm_token_type_ids=torch.zeros_like(gi), use_cache=False)
    print("B train-fwd logits_finite=%s  max|logit|=%.1f" % (
        fin(go.logits), go.logits.float().abs().max().item() if fin(go.logits) else float("nan")))

    ce = F.cross_entropy(go.logits[:, :-1, :].float().reshape(-1, go.logits.size(-1)),
                         gl[:, 1:].reshape(-1), ignore_index=-100)
    gold_lp = seq_logprob(go.logits, gi, gl)
    kl = kl_anchor(router_modules, orig_proj)
    print("  ce_finite=%s (%.4f)  gold_lp_finite=%s (%.4f)  kl_finite=%s (%.6f)" % (
        fin(ce), float(ce), fin(gold_lp), float(gold_lp), fin(kl), float(kl)))

    lo = model(input_ids=li, attention_mask=torch.ones_like(li),
               mm_token_type_ids=torch.zeros_like(li), use_cache=False)
    loop_lp = seq_logprob(lo.logits, li, ll)
    marg = 2.0 * (gold_lp - loop_lp)
    tp = -F.logsigmoid(marg)
    print("  loop_lp_finite=%s (%.4f)  marg=%.4f  tp_finite=%s (%.4f)" % (
        fin(loop_lp), float(loop_lp), float(marg), fin(tp), float(tp)))

    loss = (1.0 * ce + 0.1 * kl + 1.0 * tp) / 8
    print("  loss_finite=%s (%.4f)" % (fin(loss), float(loss)))

    # ---- per-term backward isolation: which term NaNs layer-0 router.proj? ----
    L0 = "model.language_model.layers.0.router.proj.weight"
    for tag, term in [("ce", ce), ("kl", 0.1 * kl), ("tp", tp)]:
        model.zero_grad(set_to_none=True)
        term.backward(retain_graph=True)
        bad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()]
        l0 = next((p for n, p in model.named_parameters() if n == L0), None)
        l0g = "none" if (l0 is None or l0.grad is None) else (
            "finite" if torch.isfinite(l0.grad).all() else "NaN/inf")
        print("TERM %-3s nonfinite_router_grads=%d  layer0.proj.grad=%s  sample=%s" % (
            tag, len(bad), l0g, bad[:3]))
    for h in hooks:
        h.remove()
    print("DIAG_DONE")


if __name__ == "__main__":
    main()
