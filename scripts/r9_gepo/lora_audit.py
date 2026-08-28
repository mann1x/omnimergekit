"""Audit which modules/layers the GEPO LoRA actually touched.

Reads the adapter tensor keys (ground truth) rather than trusting the config
regex, and cross-checks against the BASE model's real module inventory to find
what was reachable but never targeted.
"""
import collections
import json
import os
import re

ADAPTER = "/mnt/sdc/ml/brevity/gepo/run2/adapter_model.safetensors"
BASE = "/mnt/sdc/ream-work/armJ"

from safetensors import safe_open

keys = []
with safe_open(ADAPTER, framework="pt") as f:
    keys = list(f.keys())

print(f"=== adapter tensors: {len(keys)} ===")

lay_re = re.compile(r"model\.layers\.(\d+)\.(.+?)\.lora_[AB]")
mods = collections.Counter()
layers = collections.defaultdict(set)
for k in keys:
    m = lay_re.search(k)
    if m:
        li, mod = int(m.group(1)), m.group(2)
        mods[mod] += 1
        layers[mod].add(li)

print(f"{'module':<46} {'tensors':>8} {'layers':>7}  {'layer range'}")
print("-" * 92)
for mod, cnt in sorted(mods.items()):
    ls = sorted(layers[mod])
    rng = f"{ls[0]}..{ls[-1]}" if ls else "-"
    print(f"{mod:<46} {cnt:>8} {len(ls):>7}  {rng}")

touched = set()
for s in layers.values():
    touched |= s
print(f"\nDISTINCT LAYERS TOUCHED BY ANY LoRA: {len(touched)}")

# ---- base model inventory ----
cfg = json.load(open(os.path.join(BASE, "config.json")))
nl = cfg.get("num_hidden_layers") or cfg.get("num_layers")
print(f"\n=== base config ===")
for k in ("num_hidden_layers", "num_experts", "num_experts_per_tok",
          "shared_expert_intermediate_size", "decoder_sparse_step",
          "linear_attn_key_head_dim", "architectures"):
    if k in cfg:
        print(f"  {k}: {cfg[k]}")
lt = cfg.get("layer_types") or cfg.get("layers_block_type")
if lt:
    print(f"  layer_types: {collections.Counter(lt)}")

print(f"\n  num_hidden_layers = {nl}   LoRA touched {len(touched)}  "
      f"-> UNTOUCHED: {nl - len(touched)}" if nl else "")
if nl:
    missing = sorted(set(range(nl)) - touched)
    if missing:
        print(f"  layers with NO LoRA at all: {missing}")

# ---- what exists in the base but was never targeted ----
idx = os.path.join(BASE, "model.safetensors.index.json")
if os.path.isfile(idx):
    wm = json.load(open(idx))["weight_map"]
    base_mods = collections.Counter()
    for k in wm:
        m = re.search(r"model\.layers\.\d+\.(.+?)\.weight$", k)
        if m:
            nm = re.sub(r"experts\.\d+", "experts.N", m.group(1))
            base_mods[nm] += 1
    targeted = set(mods.keys())
    print(f"\n=== base module families: TARGETED vs NOT ===")
    print(f"{'module family':<46} {'count':>8}  {'LoRA?'}")
    print("-" * 70)
    for nm, c in sorted(base_mods.items(), key=lambda x: -x[1]):
        hit = "YES" if nm in targeted else "-- NO --"
        print(f"{nm:<46} {c:>8}  {hit}")
