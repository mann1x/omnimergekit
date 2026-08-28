#!/usr/bin/env python
"""Merge the R9 GEPO adapter into armJ and restore everything transformers drops.

TWO THINGS TRANSFORMERS SILENTLY DESTROYS HERE, BOTH FATAL LATER
----------------------------------------------------------------
1. mtp.*  -- armJ's index carries 19 MTP tensors in mtp.safetensors. Qwen3_5MoeForCausalLM
   has no such parameters, so from_pretrained ignores them and save_pretrained writes an
   index without them. config.mtp_num_hidden_layers=1 makes convert_hf_to_gguf emit
   block_count=41, so the GGUF then DECLARES 41 blocks while SHIPPING 40. llama-quantize
   builds no graph and does not notice; llama-imatrix and llama-server both die on
   'missing tensor blk.40.attn_norm.weight'. That is the 2026-08-19 armJ failure
   (ream-work/graft_mtp.py) and it is why the graft is a gate here, not a nicety.
2. config.json -- the saved config is re-serialised from the config CLASS, so any key the
   class does not model (mtp_num_hidden_layers, mtp_use_dedicated_embeddings) can vanish.
   A dropped mtp_* key changes block_count and therefore the GGUF shape. The LoRA changes
   no shape and no architecture, so the ORIGINAL config is by definition the correct one:
   copy it back verbatim rather than trusting the round-trip.

The adapter touches only self_attn.[qkvo]_proj, linear_attn.*, and mlp.shared_expert.* on
model.layers.N -- never mtp.*, never a routed expert. So armJ's mtp.safetensors is bit-
exact-correct for the merged model and is copied, not recomputed.
"""
import json
import os
import shutil
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE, ADAPTER, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
EXPECT_MTP = 19
EXPECT_TENSORS = 22712


def base_index():
    return json.load(open(os.path.join(BASE, "model.safetensors.index.json")))["weight_map"]


def merge():
    print(f">>> loading base {BASE} (cpu, bf16)", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map="cpu")
    print(f">>> applying adapter {ADAPTER}", flush=True)
    model = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT, max_shard_size="4GB", safe_serialization=True)
    tok.save_pretrained(OUT)
    print(">>> MERGE_SAVED", flush=True)


def restore_sidecars():
    """config.json is authoritative from the BASE -- see the module docstring."""
    for f in ("config.json", "generation_config.json", "chat_template.jinja",
              "preprocessor_config.json", "processor_config.json"):
        s = os.path.join(BASE, f)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(OUT, f))
            print(f"    restored {f} from base", flush=True)


def graft_mtp():
    """Copy armJ's mtp.safetensors verbatim and register its keys in the merged index."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    idx_src = base_index()
    keys = sorted(k for k in idx_src if k.startswith("mtp."))
    if len(keys) != EXPECT_MTP:
        raise SystemExit(f"FAIL: base has {len(keys)} mtp keys, expected {EXPECT_MTP}")
    mtp = {}
    for k in keys:
        with safe_open(os.path.join(BASE, idx_src[k]), framework="pt") as f:
            mtp[k] = f.get_tensor(k)

    shard = "mtp.safetensors"
    save_file(mtp, os.path.join(OUT, shard), metadata={"format": "pt"})
    ipath = os.path.join(OUT, "model.safetensors.index.json")
    idx = json.load(open(ipath))
    for k in mtp:
        idx["weight_map"][k] = shard
    json.dump(idx, open(ipath, "w"), indent=2)

    # ARTIFACT gate: re-read from disk. Writing the file proves intent; only reading the
    # index back proves the loader will find the block. That distinction is exactly what
    # let the broken REAM arms pass for a full day.
    back = json.load(open(ipath))["weight_map"]
    bad = [k for k in mtp if back.get(k) != shard]
    if bad:
        raise SystemExit(f"FAIL: {len(bad)} mtp keys absent from index, e.g. {bad[:3]}")
    sz = os.path.getsize(os.path.join(OUT, shard)) / 1e9
    print(f">>> MTP_GRAFTED {len(mtp)} tensors ({sz:.2f} GB) -> {shard}", flush=True)
    return back


def gate(wm):
    n = len(wm)
    nmtp = sum(1 for k in wm if k.startswith("mtp."))
    cfg = json.load(open(os.path.join(OUT, "config.json")))
    print(f">>> GATE tensors={n} (base {EXPECT_TENSORS}) mtp={nmtp} "
          f"mtp_num_hidden_layers={cfg.get('mtp_num_hidden_layers')} "
          f"arch={cfg.get('architectures')}", flush=True)
    if nmtp != EXPECT_MTP:
        raise SystemExit(f"FAIL: {nmtp} mtp keys in merged index, expected {EXPECT_MTP}")
    if n != EXPECT_TENSORS:
        raise SystemExit(f"FAIL: {n} tensors in merged index, base has {EXPECT_TENSORS}")
    if cfg.get("mtp_num_hidden_layers") != 1:
        raise SystemExit("FAIL: mtp_num_hidden_layers lost from config.json")
    for f in ("config.json", "chat_template.jinja", "model.safetensors.index.json"):
        if not os.path.exists(os.path.join(OUT, f)):
            raise SystemExit(f"FAIL: {f} missing from {OUT}")
    print(">>> MERGE_GATE_OK", flush=True)


if __name__ == "__main__":
    merge()
    restore_sidecars()
    gate(graft_mtp())
    print(">>> GEPO1_MERGE_DONE", flush=True)
