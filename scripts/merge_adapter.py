#!/usr/bin/env python3
"""Merge a LoRA adapter into the base and save a fresh bf16 HF dir for GGUF
conversion. Usage: merge_adapter.py <base_dir> <adapter_dir> <out_dir>

Env: MERGE_ADAPTER_DEVICE (default "0") — CUDA index to load onto.
"""
import json
import os
import shutil
import sys

import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_dir, adapter_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)


def _loader(bd):
    """Resolve the base's OWN architecture class instead of assuming CausalLM.

    AutoModelForCausalLM on a vision-language checkpoint (Qwen3.6/3.8-27B are
    Qwen3_5ForConditionalGeneration) loads the text tower only, so save_pretrained
    then writes a model with the VISION TOWER MISSING -- silently, with no error, and
    the loss of ~1.5 GB of tensors is only visible if you diff the tensor set against
    the base afterwards. Read architectures[0] and use it when transformers exposes it.
    [[feedback_dern_fold_drops_mtp_vision]]
    """
    try:
        arch = json.loads(open(os.path.join(bd, "config.json")).read())["architectures"][0]
    except Exception:
        return AutoModelForCausalLM
    cls = getattr(transformers, arch, None)
    if cls is None:
        print(f"WARNING: transformers has no {arch}; falling back to AutoModelForCausalLM. "
              "If this base is multimodal, VERIFY the output tensor set against the base.",
              flush=True)
        return AutoModelForCausalLM
    print(f"loader: {arch} (from config.architectures)", flush=True)
    return cls


base = _loader(base_dir).from_pretrained(
    base_dir, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    device_map={"": int(os.environ.get("MERGE_ADAPTER_DEVICE", "0"))})
model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
os.makedirs(out_dir, exist_ok=True)
model.save_pretrained(out_dir, max_shard_size="10GB", safe_serialization=True)
tok.save_pretrained(out_dir)
for f in ("preprocessor_config.json", "processor_config.json",
          "chat_template.jinja", "generation_config.json"):
    s = os.path.join(base_dir, f)
    if os.path.exists(s) and not os.path.exists(os.path.join(out_dir, f)):
        shutil.copy(s, out_dir)
# TENSOR-SET GATE. A merged adapter must change VALUES, never the tensor INVENTORY.
# If the loader dropped a tower, the shard index is the only place it shows.
def _names(d):
    i = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(i):
        return set(json.load(open(i))["weight_map"])
    from safetensors import safe_open
    n = set()
    for f in sorted(x for x in os.listdir(d) if x.endswith(".safetensors")):
        with safe_open(os.path.join(d, f), framework="pt") as h:
            n |= set(h.keys())
    return n


bn, on = _names(base_dir), _names(out_dir)
print(f"tensors: base={len(bn)} out={len(on)}")
if bn != on:
    miss, extra = sorted(bn - on), sorted(on - bn)
    print(f"  MISSING from output ({len(miss)}): {miss[:8]}")
    print(f"  EXTRA in output   ({len(extra)}): {extra[:8]}")
    sys.exit("MERGE_ADAPTER_FAIL: output tensor set differs from base — a tower was "
             "dropped or renamed. Refusing to report success.")
print("MERGED_OK", out_dir)
