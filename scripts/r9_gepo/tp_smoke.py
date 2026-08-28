"""Can PEFT reach the fused MoE experts / router on this stack? META-DEVICE only.

WHY meta: the audit question is structural (does a gradient path exist, and what
does it cost), not numerical. Instantiating on meta allocates NO storage and
touches NO GPU, so this runs while GPU1 is busy with the LCB repeat draw.

The claim under test, from gepo_brevity.py:75-77 -- "the 184 routed experts are
fused grouped-GEMM params, not nn.Linear, so they are un-LoRA-able by
construction". The nn.Parameter part is VERIFIED true. What is untested is
whether PEFT's `target_parameters` (present in LoraConfig on peft 0.20.0, added
for exactly this fused-MoE case) provides the route the comment says does not
exist.

Prints trainable-param cost, because "reachable" is worthless if r=32 on a
[184, 2*512, hidden] parameter costs more than the model.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from transformers import AutoConfig, AutoModelForCausalLM

BASE = "/mnt/sdc/ream-work/armJ"

import peft
from peft import LoraConfig, get_peft_model
print(f"peft={peft.__version__} torch={torch.__version__}")

cfg = AutoConfig.from_pretrained(BASE, trust_remote_code=True)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

# ---- what the fused params actually look like ----
print("\n=== fused MoE parameter shapes (layer 0) ===")
for n, p in model.named_parameters():
    if "layers.0." in n and (".experts." in n or n.endswith("mlp.gate.weight")
                             or "shared_expert_gate" in n):
        print(f"  {n:<58} {tuple(p.shape)}  numel={p.numel():,}")

base_total = sum(p.numel() for p in model.parameters())
print(f"\nbase params: {base_total:,}")

ATTEMPTS = [
    ("router only", None, ["mlp.gate.weight"]),
    ("experts only", None, ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]),
    ("current GEPO scope (control)",
     r"model\.layers\.\d+\.(self_attn\.[qkvo]_proj"
     r"|linear_attn\.(in_proj_(qkv|a|b|z)|out_proj)"
     r"|mlp\.shared_expert\.(gate|up|down)_proj)", None),
]

for label, tmod, tparam in ATTEMPTS:
    print(f"\n=== {label} ===")
    # ParamWrapper (the target_parameters path) refuses lora_dropout != 0,
    # so the fused-param arms must run at dropout 0. GEPO used 0.05 -- that is
    # a real recipe constraint, recorded here, not a smoke-test artefact.
    kw = dict(r=32, lora_alpha=64, bias="none", task_type="CAUSAL_LM",
              lora_dropout=(0.0 if tparam else 0.05))
    if tmod:
        kw["target_modules"] = tmod
    if tparam:
        kw["target_parameters"] = tparam
        kw.setdefault("target_modules", [])
    try:
        with torch.device("meta"):
            m2 = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
            pm = get_peft_model(m2, LoraConfig(**kw))
        n_tr = sum(p.numel() for p in pm.parameters() if p.requires_grad)
        n_mod = sum(1 for n, _ in pm.named_parameters()
                    if ".lora_" in n or "lora_" in n.split(".")[-1])
        print(f"  WRAPPED OK   trainable={n_tr:,}  "
              f"({100*n_tr/base_total:.3f}% of base)  lora tensors={n_mod}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")
