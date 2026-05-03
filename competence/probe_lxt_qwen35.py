#!/usr/bin/env python3
"""
Smoke-test lxt + the patched eX-LRP flow on Qwen3.5-4B.

Verifies:
  - lxt.efficient.monkey_patch with our qwen3_5 attnLRP dict succeeds.
  - The patched model still forwards on text input.
  - A backward pass produces non-zero gradients on patched modules.
  - A simple LRP-like score accumulation pass runs without erroring out.
"""
import sys
import traceback
from collections import Counter

import torch


def main():
    model_path = "/workspace/hf_models_4b/Qwen3.5-4B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 60)
    print(f"[probe2] lxt.efficient × qwen3_5 smoke test")
    print(f"[probe2] model: {model_path}")
    print(f"[probe2] device: {device}, dtype: {dtype}")
    print("=" * 60)

    # 1. lxt.efficient API
    try:
        from lxt.efficient import monkey_patch
        from lxt.efficient.models.qwen3_5 import attnLRP
        from transformers.models.qwen3_5 import modeling_qwen3_5
        print(f"[probe2] qwen3_5 attnLRP entries: {len(attnLRP)}")
    except Exception:
        print("[FAIL] lxt.efficient or qwen3_5 import failed:")
        traceback.print_exc()
        return 1

    # 2. Monkey-patch BEFORE loading the model. Signature is
    #    monkey_patch(module, patch_map=None, verbose=False) — first arg is the
    #    transformers modeling module to mutate, second is the patch dict.
    print("[probe2] applying monkey_patch(modeling_qwen3_5, attnLRP)...")
    try:
        monkey_patch(modeling_qwen3_5, attnLRP, verbose=True)
    except Exception:
        print("[FAIL] monkey_patch raised:")
        traceback.print_exc()
        return 2

    # 3. Load model.
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device,
            attn_implementation="eager",
        )
        model.eval()
        print(f"[probe2] loaded {type(model).__name__}")
    except Exception:
        print("[FAIL] model load failed:")
        traceback.print_exc()
        return 3

    # 4. Forward.
    try:
        ids = tok("def fibonacci(n):", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(input_ids=ids)
        print(f"[probe2] forward OK. logits shape = {tuple(out.logits.shape)}")
    except Exception:
        print("[FAIL] forward failed:")
        traceback.print_exc()
        return 4

    # 5. Backward — this is the LRP-style relevance propagation.
    try:
        for p in model.parameters():
            p.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        ids = tok("def fibonacci(n):", return_tensors="pt").input_ids.to(device)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        n_grads = sum(1 for p in model.parameters() if p.grad is not None)
        # bucket grad magnitudes by tensor-name category
        buckets = Counter()
        bucket_norms = {}
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            cat = "other"
            if "mlp.gate_proj" in n: cat = "mlp.gate_proj"
            elif "mlp.up_proj" in n: cat = "mlp.up_proj"
            elif "mlp.down_proj" in n: cat = "mlp.down_proj"
            elif "self_attn.q_proj" in n: cat = "attn.q_proj"
            elif "self_attn.k_proj" in n: cat = "attn.k_proj"
            elif "self_attn.v_proj" in n: cat = "attn.v_proj"
            elif "self_attn.o_proj" in n: cat = "attn.o_proj"
            elif "linear_attn" in n: cat = "linear_attn"
            elif "norm" in n: cat = "norm"
            elif "embed_tokens" in n: cat = "embed_tokens"
            elif "lm_head" in n: cat = "lm_head"
            buckets[cat] += 1
            bucket_norms[cat] = bucket_norms.get(cat, 0.0) + float(p.grad.abs().sum())
        print(f"[probe2] backward OK. params with grads = {n_grads}")
        print("[probe2] grad |·|_1 per category (LRP relevance proxy):")
        for cat in sorted(buckets, key=lambda c: -bucket_norms.get(c, 0)):
            print(f"    {cat:20s} n={buckets[cat]:3d}  Σ|g|={bucket_norms.get(cat, 0):.4e}")
    except Exception:
        print("[FAIL] backward failed:")
        traceback.print_exc()
        return 5

    print()
    print("=" * 60)
    print("[probe2] SUMMARY")
    print("[probe2] - monkey_patch(qwen3_5.attnLRP): ✓")
    print("[probe2] - patched forward + backward: ✓")
    print("[probe2] - per-category grad magnitudes available for LRP scoring")
    print("[probe2] If grads are nonzero across mlp/attn/linear_attn, eX-LRP")
    print("[probe2] scoring will work end-to-end on Qwen3.5-4B.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
