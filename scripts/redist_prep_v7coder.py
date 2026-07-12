#!/usr/bin/env python3
"""redist_prep_v7coder.py — T203 prep (CPU, no model load beyond shard headers).

1. Verify the v7-coder 98e student has NO residual pruned-expert index (message-3
   model hazard): expert stacks, router out-dim, per_expert_scale width must all be
   98, never 128.
2. Build keep_meta {keep, drop, num_experts} from the C6v3lcb drop map for the
   HCSMoE fit (keep[li] = 0..127 minus drop[li]).

Usage: redist_prep_v7coder.py <student_dir> <drop_map.json> <out_keepmeta.json>
"""
import json
import os
import sys

from safetensors import safe_open

STUDENT = sys.argv[1]
DROP_MAP = sys.argv[2]
OUT = sys.argv[3]

idx = json.load(open(os.path.join(STUDENT, "model.safetensors.index.json")))["weight_map"]


def shape_of(name):
    shard = idx.get(name)
    if shard is None:
        return None
    with safe_open(os.path.join(STUDENT, shard), framework="pt") as f:
        return list(f.get_slice(name).get_shape())


print("=== expert / router / per_expert_scale shapes (layer 0) ===")
probe = [
    "model.language_model.layers.0.experts.gate_up_proj",
    "model.language_model.layers.0.experts.down_proj",
]
# discover router + per_expert_scale tensor names in layer 0
l0 = [k for k in idx if k.startswith("model.language_model.layers.0.")]
router_names = [k for k in l0 if "router" in k or "gate" in k.lower() and "experts" not in k]
pes_names = [k for k in l0 if "per_expert_scale" in k or "expert_scale" in k]
for n in probe + sorted(set(router_names + pes_names)):
    print(f"  {n.split('layers.0.')[-1]:40s} {shape_of(n)}")

# expert-dim sanity across a few layers
print("\n=== expert dim (axis 0 of gate_up_proj) per sampled layer ===")
bad = []
for li in [0, 7, 15, 22, 29]:
    s = shape_of(f"model.language_model.layers.{li}.experts.gate_up_proj")
    e = s[0] if s else None
    flag = "" if e == 98 else "  <-- NOT 98!"
    if e != 98:
        bad.append((li, e))
    print(f"  L{li:2d} experts={e}{flag}")
print("RESIDUAL-PRUNED-INDEX CHECK:", "CLEAN (all 98)" if not bad else f"PROBLEM {bad}")

# build keep_meta from drop map
drop_raw = json.load(open(DROP_MAP))
drop = {int(k): sorted(int(x) for x in v) for k, v in drop_raw.items()}
NUM_E = 128
keep = {li: sorted(set(range(NUM_E)) - set(drop[li])) for li in drop}
nbad = {li: len(v) for li, v in drop.items() if len(v) != 30}
nkeep = {li: len(v) for li, v in keep.items() if len(v) != 98}
print("\n=== keep_meta from C6v3lcb drop map ===")
print(f"  layers: {len(drop)}  drop/layer!=30: {nbad or 'none'}  keep/layer!=98: {nkeep or 'none'}")
keep_meta = {"keep": {str(li): keep[li] for li in keep},
             "drop": {str(li): drop[li] for li in drop},
             "num_experts": NUM_E}
json.dump(keep_meta, open(OUT, "w"))
print(f"  wrote {OUT}")

# peek expert_drop_metadata.json
edm = os.path.join(STUDENT, "expert_drop_metadata.json")
if os.path.exists(edm):
    d = json.load(open(edm))
    print(f"\n=== expert_drop_metadata.json keys: {list(d.keys())[:12]} ===")
