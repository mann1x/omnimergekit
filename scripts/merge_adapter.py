#!/usr/bin/env python3
"""Merge a LoRA adapter into the base and save a fresh bf16 HF dir for GGUF
conversion. Usage: merge_adapter.py <base_dir> <adapter_dir> <out_dir>"""
import os
import shutil
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_dir, adapter_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    base_dir, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    device_map={"": 0})
model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
os.makedirs(out_dir, exist_ok=True)
model.save_pretrained(out_dir, max_shard_size="10GB", safe_serialization=True)
tok.save_pretrained(out_dir)
for f in ("preprocessor_config.json", "processor_config.json",
          "chat_template.jinja", "generation_config.json"):
    s = os.path.join(base_dir, f)
    if os.path.exists(s) and not os.path.exists(os.path.join(out_dir, f)):
        shutil.copy(s, out_dir)
print("MERGED_OK", out_dir)
