"""Verify the patched scope logic: default MUST still be run2, router adds exactly the router.

Imports the real constants from gepo_brevity so this tests the shipped code path,
not a re-typed copy of it. Meta device: no GPU, no weights, safe while GPU1 is busy.

run2's recorded trainable_params = 42,332,160 (run_meta.json). If the default arm
does not reproduce that number exactly, the patch broke run1/run2 reproducibility
and must not ship.
"""
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from gepo_brevity import (LORA_REGEX, N_LORA_TARGETS, ROUTER_PARAMS,
                          ROUTER_PARAM_RE, N_ROUTER_TARGETS)

BASE = "/mnt/sdc/ream-work/armJ"
RUN2_TRAINABLE = 42_332_160
ROUTER_EXPECTED = 2_856_960

cfg = AutoConfig.from_pretrained(BASE, trust_remote_code=True)
with torch.device("meta"):
    probe = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

import re
nlin = sum(1 for n, m in probe.named_modules()
           if isinstance(m, torch.nn.Linear) and re.fullmatch(LORA_REGEX, n))
nrt = sum(1 for n, _ in probe.named_parameters() if ROUTER_PARAM_RE.fullmatch(n))
print(f"gate inputs: linear match={nlin} (expect {N_LORA_TARGETS})  "
      f"router match={nrt} (expect {N_ROUTER_TARGETS})")
assert nlin == N_LORA_TARGETS, "LINEAR GATE would refuse"
assert nrt == N_ROUTER_TARGETS, "ROUTER GATE would refuse"


def build(router: bool):
    kw = dict(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
              task_type="CAUSAL_LM", target_modules=LORA_REGEX)
    if router:                      # mirrors gepo_brevity.py exactly
        kw["lora_dropout"] = 0.0
        kw["target_parameters"] = ROUTER_PARAMS
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        pm = get_peft_model(m, LoraConfig(**kw))
    return sum(p.numel() for p in pm.parameters() if p.requires_grad)


base = build(False)
rout = build(True)
print(f"\ndefault (run1/run2 path) trainable = {base:,}   expect {RUN2_TRAINABLE:,}"
      f"   {'OK' if base == RUN2_TRAINABLE else 'MISMATCH -- DO NOT SHIP'}")
print(f"--router-lora            trainable = {rout:,}")
print(f"delta                               = {rout-base:,}   expect "
      f"{ROUTER_EXPECTED:,}   "
      f"{'OK' if rout-base == ROUTER_EXPECTED else 'MISMATCH'}")
ok = (base == RUN2_TRAINABLE) and (rout - base == ROUTER_EXPECTED)
print("\nSCOPE_VERIFY_OK" if ok else "\nSCOPE_VERIFY_FAIL")
sys.exit(0 if ok else 1)
