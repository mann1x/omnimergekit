#!/usr/bin/env python3
"""Does one forward pass of this model, repeated on identical input, give identical logprobs?

WHY THIS EXISTS
---------------
The 2026-09-09 GRPO smoke reported, at step 1, BEFORE any optimiser update, with
LoRA `B=0` and `lora_dropout=0.0`:

    kl = 0.3166          policy vs the ADAPTER-DISABLED reference
    sampling/sampling_logp_difference/mean = 1.128   trainer vs vLLM

The first number should be EXACTLY ZERO. With `B=0` the adapter contributes nothing,
so "policy" and "reference" are the same weights; TRL's k3 estimator
`exp(d) - d - 1` is quadratic and non-negative, so kl=0.3166 implies a mean per-token
|d| of roughly 0.8 nats between two forward passes of an identical model.

That is suspiciously close to the 0.92-1.13 nats measured against vLLM. If BOTH
comparisons sit near ~1 nat, then vLLM is not the variable -- the model simply does not
reproduce its own logprobs -- and the right response is to size the importance-sampling
correction for it, not to hunt an engine mismatch.

Jprime-p3 is a 128-expert top-8 MoE. The mechanism to look for is ROUTING FLIPS: a
difference far below 1e-3 in a router logit changes WHICH experts a token uses, and the
token's logprob then changes categorically rather than drifting. That predicts a
heavy-tailed difference -- most tokens near-identical, a minority enormous -- which is
exactly the shape of `mean 1.128 / max 37.64`. A dense model cannot produce that.

RESOLVED 2026-09-09 -- READ THIS BEFORE USING THE PROBE
------------------------------------------------------
The hypothesis above is REFUTED. `kl=0.3166` was not the model failing to reproduce
itself; it was the LIGER FUSED LOSS computing a wrong KL. With `--liger` removed and
nothing else changed, the same step reports:

    kl = 0            EXACTLY zero, as B=0 demands
    grad_norm = 0.02768   (was nan; grads nonfinite 0/410, was 410/410)

So two forward passes of these weights DO agree, and the MoE-routing-flip story does not
apply to the policy-vs-reference gap. What survived the control is the TRAINER-vs-vLLM
gap, `sampling_logp_difference/mean = 1.128`, byte-identical with and without Liger --
that one is real, cross-engine, and is what the importance-sampling correction is for.

The probe is kept because "does this model reproduce its own logprobs" is worth being
able to answer in one command for any future arm -- but it is NOT needed to settle the
2026-09-09 question, which the Liger A/B already closed.

WHAT IT MEASURES (no training loop, no vLLM, no TRL)
----------------------------------------------------
Same weights, same input, two forwards. Reports the per-token logprob difference
distribution. Three arms, because they isolate different claims:

  eval_twice   model.eval(), two passes            -- pure kernel/routing determinism
  train_twice  model.train(), two passes           -- adds anything train-mode changes
  adapter      PEFT attached, B=0: enabled vs disabled_adapter()
               -- reproduces TRL's ACTUAL policy-vs-reference comparison

INTERPRETATION
--------------
  all three ~0 (< 1e-6)   -> the model is deterministic; kl=0.3166 has another source
                             and the MoE-routing hypothesis is REFUTED. Look at how TRL
                             builds the reference pass instead.
  eval_twice ~1 nat       -> CONFIRMED: the model does not reproduce itself. The logp
                             gap is intrinsic, not a vLLM problem.
  eval_twice ~0 but
  adapter   ~1 nat        -> determinism is fine; the adapter enable/disable round trip
                             is what perturbs it. That is a PEFT issue, and actionable.

A heavy tail is the signature that matters, so p50/p99/max are printed, not just a mean.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch


def summarise(tag: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    """Per-token |logprob difference| distribution between two passes."""
    d = (a.float() - b.float()).abs().flatten()
    n = d.numel()
    q = torch.tensor([0.5, 0.9, 0.99], device=d.device)
    p50, p90, p99 = torch.quantile(d.float(), q).tolist() if n < 16_000_000 else (
        float("nan"), float("nan"), float("nan"))
    out = {
        "arm": tag,
        "n_tokens": int(n),
        "mean": d.mean().item(),
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "max": d.max().item(),
        "frac_gt_0.01": (d > 0.01).float().mean().item(),
        "frac_gt_1.0": (d > 1.0).float().mean().item(),
        "bit_identical": bool(torch.equal(a, b)),
    }
    print(f">>> {tag:12s} mean={out['mean']:.6f} p50={out['p50']:.2e} "
          f"p99={out['p99']:.4f} max={out['max']:.4f} "
          f">0.01={out['frac_gt_0.01']:.4f} >1.0={out['frac_gt_1.0']:.4f} "
          f"bit_identical={out['bit_identical']}", flush=True)
    return out


def logprobs(model, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """Per-token logprob of the ACTUAL next token -- the same quantity TRL compares.

    mm_token_type_ids is passed explicitly: transformers >= 5.5.0 Gemma-4 forward()
    REQUIRES it in train mode and raises without it. All-zeros means "text only".
    """
    kw = {"input_ids": ids, "attention_mask": attn, "use_cache": False}
    if "gemma" in type(model).__name__.lower() or hasattr(model, "language_model"):
        kw["mm_token_type_ids"] = torch.zeros_like(ids)
    try:
        logits = model(**kw).logits
    except (TypeError, ValueError):
        kw.pop("mm_token_type_ids", None)
        logits = model(**kw).logits
    logits = logits[:, :-1, :]
    tgt = ids[:, 1:]
    # gather in fp32: a bf16 log_softmax over a 262k vocab loses the precision this
    # probe is trying to measure.
    lp = torch.log_softmax(logits.float(), dim=-1)
    return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-adapter", action="store_true")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    print(f">>> loading {a.model} (bfloat16)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda:0")

    vocab = int(getattr(model.config, "vocab_size", tok.vocab_size))
    ids = torch.randint(0, vocab, (a.batch, a.seq_len), device="cuda:0")
    attn = torch.ones_like(ids)
    print(f">>> batch={a.batch} seq_len={a.seq_len} vocab={vocab}", flush=True)

    results = []

    model.eval()
    with torch.no_grad():
        results.append(summarise("eval_twice",
                                 logprobs(model, ids, attn),
                                 logprobs(model, ids, attn)))

    model.train()
    with torch.no_grad():
        results.append(summarise("train_twice",
                                 logprobs(model, ids, attn),
                                 logprobs(model, ids, attn)))

    if not a.skip_adapter:
        # Reproduce TRL's policy-vs-reference comparison exactly: PEFT attached with
        # B=0 (peft's default init), reference obtained by DISABLING the adapter. With
        # B=0 these are the same function, so any difference is machinery, not maths.
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0,
                         bias="none", task_type="CAUSAL_LM",
                         target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj)$")
        pm = get_peft_model(model, cfg)
        pm.eval()
        with torch.no_grad():
            pol = logprobs(pm, ids, attn)
            with pm.disable_adapter():
                ref = logprobs(pm, ids, attn)
        results.append(summarise("adapter_B0", pol, ref))

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"model": a.model, "batch": a.batch, "seq_len": a.seq_len,
                       "seed": a.seed, "arms": results}, fh, indent=2)
        print(f">>> wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
