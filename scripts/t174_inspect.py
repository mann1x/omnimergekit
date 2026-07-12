#!/usr/bin/env python3
"""T174.1b: dump Gemma-4 MoE module tree so we can scope LoRA correctly.
Gemma4ClippableLinear wraps an inner .linear (real nn.Linear); PEFT must target
the inner linears, scoped to the language model only (not vision/experts/router).
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

A2 = "/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it"
m = AutoModelForCausalLM.from_pretrained(
    A2, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    attn_implementation="eager", device_map={"": 0})

print("TOP:", type(m).__name__, "children=", [n for n, _ in m.named_children()])
for n, c in m.named_children():
    print("  ", n, "->", [cn for cn, _ in c.named_children()][:8])


def first_path(substr, after=None):
    for name, mod in m.named_modules():
        if substr in name and (after is None or after in name):
            return name
    return None


# locate a text decoder layer 0 and a vision layer 0
def dump_leaves(prefix, label):
    print(f"\n=== {label}  (prefix='{prefix}') ===")
    seen = 0
    for name, mod in m.named_modules():
        if name.startswith(prefix) and len(list(mod.children())) == 0:
            t = type(mod).__name__
            print(f"  {name[len(prefix):]:55s} {t}")
            seen += 1
            if seen > 40:
                print("  ...")
                break


# find language-model layer-0 prefix and vision layer-0 prefix
lm0 = None
vis0 = None
exp = None
rtr = None
for name, mod in m.named_modules():
    if lm0 is None and name.endswith("layers.0") and "vision" not in name and "vision_tower" not in name:
        lm0 = name
    if vis0 is None and "vision" in name and name.endswith("layers.0"):
        vis0 = name
    if exp is None and name.endswith(".experts"):
        exp = name
    if rtr is None and name.endswith(".router"):
        rtr = name
print("\nlm0 =", lm0, "\nvis0=", vis0, "\nexperts=", exp, "\nrouter=", rtr)
if lm0:
    dump_leaves(lm0 + ".", "TEXT LAYER 0 leaves")
if vis0:
    dump_leaves(vis0 + ".", "VISION LAYER 0 leaves")

# how many real nn.Linear named '*.linear' under language model vs vision
lm_lin = vis_lin = other_lin = 0
for name, mod in m.named_modules():
    if isinstance(mod, nn.Linear):
        if "vision" in name:
            vis_lin += 1
        elif "language_model" in name or (lm0 and lm0.rsplit(".layers", 1)[0] in name):
            lm_lin += 1
        else:
            other_lin += 1
print(f"\nnn.Linear counts: language_model={lm_lin}  vision={vis_lin}  other={other_lin}")

# show the inner structure of one text q_proj and one mlp gate_proj
for tag in ["self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"]:
    p = first_path(lm0 + "." + tag) if lm0 else None
    if p:
        mod = dict(m.named_modules())[p]
        kids = [(cn, type(cm).__name__) for cn, cm in mod.named_children()]
        print(f"\n{tag}: {type(mod).__name__}  children={kids}")
