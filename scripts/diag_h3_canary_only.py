#!/usr/bin/env python3
"""H3: run the canary ONLY (no training) against plain A2 with current router_kd.py code.
If this produces the 'post-EAC = 1/5 loop' pattern, the apparent recovery was always a
canary-code artifact — EAC never actually fixed anything.
"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, "/srv/ml/scripts")
import router_kd as rk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

A2 = "/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it"  # plain A2 base bytes
CANARY = Path("/srv/ml/scripts/ifeval_rumination_canaries.json")
MAX_NEW = 4096  # script default

print(f"=== H3: canary on plain A2 with CURRENT router_kd.py code ===", flush=True)
print(f"variant: {A2}")
print(f"canary : {CANARY}")
print(f"max_new: {MAX_NEW}")
print(f"router_kd.py mtime: 19:26 UTC (after vanilla RKD 18:43, before iterate 20:46)", flush=True)
print()
print("Loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(A2)
print("Loading model bf16 on cuda:0...", flush=True)
model = AutoModelForCausalLM.from_pretrained(A2, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
model.eval()
print("Running canary...", flush=True)
rows = rk.run_canary(tok, model, CANARY, MAX_NEW, "H3-test")
print()
print("=== H3 RESULT ===")
loopers = sum(1 for r in rows if r.get("looped"))
print(f"loopers: {loopers}/{len(rows)}")
print()
print("EXPECTED IF EAC IS A NO-OP (council recipe pre-EAC at 20:46): 1/5 looper, chars 41-1845")
print("EXPECTED IF EAC IS NOT A NO-OP (vanilla RKD at 18:43):         3/5 loopers, chars 1616-13926")
