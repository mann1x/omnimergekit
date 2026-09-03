#!/usr/bin/env python3
"""Apply a LoRA adapter directly to safetensors shards. No transformers round-trip.

Usage: apply_lora_to_safetensors.py <base_dir> <adapter_dir> <out_dir>

WHY NOT merge_adapter.py (the transformers path)
------------------------------------------------
On Qwen3.6/3.8-27B (Qwen3_5ForConditionalGeneration) the load->merge_and_unload->
save_pretrained round-trip produced, measured 2026-09-02:
  * 1184 tensors instead of 1199 -- all 15 mtp.* (NextN) tensors DROPPED, and
  * every text-tower key TRIPLE-nested:
      model.language_model.language_model.language_model.embed_tokens.weight
Neither raises. A GGUF convert of that anchor would simply have produced a model with
no MTP head and unresolvable names. [[feedback_dern_fold_drops_mtp_vision]]

This script never constructs a model. It reads the shards, adds scale * (B @ A) to the
targeted weights in place, and writes every other byte through untouched, so the tensor
inventory CANNOT change: names, dtypes, shard layout and index.json are preserved.

MATH: peft LoRA, W' = W + (alpha/r) * B @ A   (rslora would use alpha/sqrt(r); refused
below if set, as would DoRA, whose magnitude column this does not implement).
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

base_dir, adapter_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
if cfg.get("use_dora"):
    sys.exit("REFUSE: use_dora=True — the magnitude column is not implemented here.")
if cfg.get("modules_to_save"):
    sys.exit(f"REFUSE: modules_to_save={cfg['modules_to_save']} — those are full-tensor "
             "replacements this script does not apply.")
r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
scale = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
print(f"LoRA r={r} alpha={alpha} rslora={bool(cfg.get('use_rslora'))} -> scale={scale}")

# ---- gather A/B pairs, keyed by the BASE tensor name they modify
pairs = {}
ad = os.path.join(adapter_dir, "adapter_model.safetensors")
with safe_open(ad, framework="pt") as f:
    for k in f.keys():
        if ".lora_A.weight" not in k and ".lora_B.weight" not in k:
            continue
        which = "A" if ".lora_A.weight" in k else "B"
        stem = k.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
        for pfx in ("base_model.model.", "base_model."):
            if stem.startswith(pfx):
                stem = stem[len(pfx):]
                break
        pairs.setdefault(stem + ".weight", {})[which] = f.get_tensor(k)
bad = [k for k, v in pairs.items() if set(v) != {"A", "B"}]
if bad:
    sys.exit(f"REFUSE: {len(bad)} modules have an unpaired A/B: {bad[:3]}")
print(f"adapter targets {len(pairs)} modules")

idx_path = os.path.join(base_dir, "model.safetensors.index.json")
index = json.load(open(idx_path))
wmap = index["weight_map"]
missing = [k for k in pairs if k not in wmap]
if missing:
    sys.exit(f"REFUSE: {len(missing)} adapter targets are absent from the base index — "
             f"name mapping is wrong: {missing[:3]}")
print(f"all {len(pairs)} targets resolve against the base index")

os.makedirs(out_dir, exist_ok=True)
by_shard = {}
for name, shard in wmap.items():
    by_shard.setdefault(shard, []).append(name)

applied = 0
for shard in sorted(by_shard):
    src = os.path.join(base_dir, shard)
    tensors, meta = {}, None
    with safe_open(src, framework="pt") as f:
        meta = f.metadata()
        for name in f.keys():
            t = f.get_tensor(name)
            if name in pairs:
                A = pairs[name]["A"].to(torch.float32)
                B = pairs[name]["B"].to(torch.float32)
                delta = (B @ A) * scale
                if delta.shape != t.shape:
                    sys.exit(f"REFUSE: {name} delta {tuple(delta.shape)} != weight "
                             f"{tuple(t.shape)}")
                t = (t.to(torch.float32) + delta).to(t.dtype)
                applied += 1
            tensors[name] = t
    save_file(tensors, os.path.join(out_dir, shard), metadata=meta or {"format": "pt"})
    print(f"  {shard}: {len(tensors)} tensors written", flush=True)

if applied != len(pairs):
    sys.exit(f"REFUSE: applied {applied} of {len(pairs)} adapter modules")

for f in os.listdir(base_dir):
    if f.endswith(".safetensors") or f == ".COMPLETE":
        continue
    s = os.path.join(base_dir, f)
    if os.path.isfile(s):
        shutil.copy2(s, os.path.join(out_dir, f))


def names(d):
    return set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"])


bn, on = names(base_dir), names(out_dir)
print(f"tensors: base={len(bn)} out={len(on)} identical_set={bn == on}")
if bn != on:
    sys.exit(f"APPLY_LORA_FAIL: inventory changed. missing={sorted(bn-on)[:5]} "
             f"extra={sorted(on-bn)[:5]}")

# A NON-TARGETED tensor must be bit-identical, and a TARGETED one must have changed.
# Without both, "identical inventory" is compatible with having written nothing.
def get(d, n):
    with safe_open(os.path.join(d, wmap[n]), framework="pt") as h:
        return h.get_tensor(n)


ctrl = next(n for n in bn if n not in pairs and n.endswith(".weight"))
tgt = next(iter(pairs))
same = torch.equal(get(base_dir, ctrl), get(out_dir, ctrl))
diff = not torch.equal(get(base_dir, tgt), get(out_dir, tgt))
print(f"  untouched {ctrl}: identical={same}")
print(f"  targeted  {tgt}: changed={diff}")
if not (same and diff):
    sys.exit("APPLY_LORA_FAIL: control checks failed")
print(f"APPLY_LORA_OK applied={applied} modules -> {out_dir}")
