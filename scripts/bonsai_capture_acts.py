#!/usr/bin/env python3
"""bonsai_capture_acts.py — capture the REAL input of one linear layer of base Qwen3.8-27B.

Builds the text model truncated to the first `layer+1` blocks (the input of a layer depends only
on the blocks before it), loads those weights from the base safetensors shards, runs real text,
and saves the tensor fed to the target module. Used as the input set of the single-layer
divergence bench, replacing the Gaussian proxy.
"""
import argparse
import glob
import json

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--module", required=True, help="e.g. self_attn.k_proj")
ap.add_argument("--text", required=True)
ap.add_argument("--n-tokens", type=int, default=8192)
ap.add_argument("--ctx", type=int, default=1024)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:1")
a = ap.parse_args()

cfg = json.load(open(f"{a.model_dir}/config.json"))
tc = cfg["text_config"]
n = a.layer + 1
tc["num_hidden_layers"] = n
for k, v in list(tc.items()):
    if isinstance(v, list) and len(v) == 64:
        tc[k] = v[:n]
config = Qwen3_5Config(**cfg).text_config if "text_config" in cfg else Qwen3_5Config(**cfg)
model = Qwen3_5ForCausalLM(config).to(torch.bfloat16)
want = dict(model.state_dict())
loaded, idx = 0, json.load(open(f"{a.model_dir}/model.safetensors.index.json"))["weight_map"]
sd = {}
for shard in sorted(set(idx.values())):
    path = f"{a.model_dir}/{shard}"
    if not glob.glob(path):
        continue
    keys = [k for k, s in idx.items() if s == shard]
    with safe_open(path, "pt") as f:
        for k in keys:
            for cand in (k, k.replace("model.language_model.", "model."), k.replace("model.language_model.", "")):
                if cand in want and cand not in sd:
                    sd[cand] = f.get_tensor(k)
                    loaded += 1
                    break
missing = [k for k in want if k not in sd and "lm_head" not in k and not k.startswith("model.norm")]
print(f"loaded {loaded} tensors; missing (excluding head/final norm): {len(missing)} {missing[:5]}", flush=True)
assert not missing, "the truncated model is not fully initialised from base weights"
model.load_state_dict(sd, strict=False)
model = model.to(a.device).eval()

target = model.model.layers[a.layer].get_submodule(a.module)
grabbed = []
target.register_forward_pre_hook(lambda m, inp: grabbed.append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu()))
tok = AutoTokenizer.from_pretrained(a.model_dir)
ids = tok(open(a.text).read()[:400000], return_tensors="pt").input_ids[0][: a.n_tokens]
with torch.no_grad():
    for i in range(0, len(ids), a.ctx):
        model.model(input_ids=ids[i:i + a.ctx][None].to(a.device))
X = torch.cat(grabbed)
print(f"captured {tuple(X.shape)}  mean|x|={X.abs().mean():.4f}  rms={X.pow(2).mean().sqrt():.4f}  "
      f"kurtosis={((X - X.mean()) ** 4).mean() / X.var() ** 2:.2f}", flush=True)
torch.save(X, a.out)
print("ACTS_OK", a.out)
