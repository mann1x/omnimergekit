#!/usr/bin/env python3
"""T174.1 pre-flight: validate the anti-loop LoRA mechanics on A2 before any
teacher-gen compute. Confirms (a) PEFT wraps ONLY language-model attn + shared
MLP (NOT vision tower, NOT packed experts, NOT router), (b) a completion-masked
forward with mm_token_type_ids gives a finite loss and grads flow to adapters.
"""
import collections

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

A2 = "/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it"
# LM projections are plain nn.Linear; vision uses Gemma4ClippableLinear (unsupported)
# and the packed experts have no leaf Linears. Scope a regex to the language-model
# attention + shared dense MLP ONLY (router.proj deliberately excluded).
TARGETS = (r"model\.language_model\.layers\.\d+\."
           r"(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)")

print("[load] A2 bf16 -> GPU0 ...", flush=True)
tok = AutoTokenizer.from_pretrained(A2, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    A2, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    attn_implementation="eager", device_map={"": 0})
model.train()

cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                 task_type="CAUSAL_LM", target_modules=TARGETS)
model = get_peft_model(model, cfg)
model.print_trainable_parameters()

cats = collections.Counter()
samples = collections.defaultdict(list)
for n, p in model.named_parameters():
    if p.requires_grad and "lora_" in n:
        base = n.split(".lora_")[0]
        if "vision" in n or "embed_vision" in n:
            c = "VISION(!!)"
        elif ".experts." in n:
            c = "EXPERTS(!!)"
        elif ".router." in n:
            c = "ROUTER(!!)"
        elif "language_model" in n or ".model.layers." in n:
            c = "LM"
        else:
            c = "OTHER:" + base.rsplit(".", 2)[0][-40:]
        cats[c] += 1
        if len(samples[c]) < 2:
            samples[c].append(base)

print("\n[wrapped-module categories]")
for c, n in cats.most_common():
    print("  %-14s %4d   e.g. %s" % (c, n, samples[c]))

bad = [c for c in cats if "(!!)" in c]
print("\nASSERT no vision/experts/router wrapped:", ("FAIL " + str(bad)) if bad else "OK")
assert sum(cats.values()) > 0, "no adapters attached!"

msgs = [{"role": "user", "content": "Write a one-sentence greeting."}]
ptxt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
pids = tok(ptxt, add_special_tokens=False)["input_ids"]
comp = tok("Hello there, nice to meet you!", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
ids = torch.tensor([pids + comp], device=model.device)
labels = torch.tensor([[-100] * len(pids) + comp], device=model.device)
attn = torch.ones_like(ids)
mm = torch.zeros_like(ids)
out = model(input_ids=ids, attention_mask=attn, mm_token_type_ids=mm, use_cache=False)
logits = out.logits[:, :-1, :].float()
tgt = labels[:, 1:]
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
print("\n[forward] completion-masked CE loss = %.4f  finite=%s" % (
    loss.item(), torch.isfinite(loss).item()))
loss.backward()
g = sum(1 for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None and "lora_" in n)
print("[backward] adapters with non-None grad = %d" % g)
print("\nPREFLIGHT_OK" if (torch.isfinite(loss).item() and not bad and g > 0) else "PREFLIGHT_FAIL")
