#!/usr/bin/env python3
"""Write a trained bonsai4b_qat.py student as an ordinary dense checkpoint that any runtime can serve.

Each ternary tensor is stored in the latent file in the rotated basis. Its served form is
unrotate(top86(latent)): the trit * group-scale weight carried back to the model's own basis, which is exactly
what the training forward computes. The trained full-precision tensors (norms, in_proj_a/b, conv1d, A_log,
dt_bias) are copied as trained. Every other tensor of the base checkpoint (MTP head, vision tower) is copied
byte-for-byte, so the result keeps the base file layout.

    bonsai4b_export.py --base Qwen3.5-4B --latent r1_mix25/latent.pt --out r1_mix25/export
"""
import argparse
import json
import math
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--latent", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:1")
a = ap.parse_args()
dev = a.device
G, NZERO = 128, 42
HM = {}


def blk(n):
    assert n % 512 == 0, n
    return 1024 if n % 1024 == 0 else 512


def hm(b):
    if b not in HM:
        h = torch.tensor([[1.0]])
        while h.shape[0] < b:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        HM[b] = (h / math.sqrt(b)).to(dev)
    return HM[b]


def signs_for(n):
    g = torch.Generator().manual_seed(1234 + n)
    return (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(dev)


def unrot(x, s):
    sh, b = x.shape, blk(x.shape[-1])
    return ((x.reshape(-1, sh[-1] // b, b) @ hm(b)) * s.reshape(1, -1, b)).reshape(sh)


def top86(w):
    o, i = w.shape
    wg = w.reshape(o, i // G, G)
    aw = wg.abs()
    mask = aw > aw.kthvalue(NZERO, dim=-1, keepdim=True).values
    s = (aw * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1)
    return (s * torch.sign(wg) * mask).reshape(o, i)


ck = torch.load(a.latent, map_location="cpu")
state, step = ck["state"], ck["step"]
new = {}                                           # base file key -> replacement tensor (float32, cpu)
for k, v in state.items():
    if k.startswith("lm_head."):                   # the tied embedding under its second name
        continue
    assert k.startswith("model."), k
    assert not k.endswith((".thr", ".scale")), f"{k}: only the strict top-86 format is exported"
    fk = "model.language_model." + k[len("model."):]
    if k.endswith(".tw.weight"):
        w = v.to(dev).float()
        new[fk.replace(".tw.weight", ".weight")] = unrot(top86(w), signs_for(w.shape[1])).cpu()
    else:
        new[fk] = v.float()
n_tern = sum(1 for k in state if k.endswith(".tw.weight") and not k.startswith("lm_head."))
print(f"latent step {step}: {n_tern} ternary tensors, {len(new) - n_tern} full-precision trained tensors")

os.makedirs(a.out, exist_ok=True)
used = set()
for fn in sorted(os.listdir(a.base)):
    src = os.path.join(a.base, fn)
    if not fn.endswith(".safetensors"):
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(a.out, fn))
        continue
    out = {}
    with safe_open(src, "pt") as f:
        meta = f.metadata()
        for k in f.keys():
            t = f.get_tensor(k)
            if k in new:
                assert new[k].shape == t.shape, (k, new[k].shape, t.shape)
                out[k] = new[k].to(t.dtype)
                used.add(k)
            else:
                out[k] = t
    save_file(out, os.path.join(a.out, fn), metadata=meta)
    print(f"{fn}: {len(out)} tensors, {sum(1 for k in out if k in new)} replaced")
    del out
missing = sorted(set(new) - used)
assert not missing, f"not found in the base files: {missing[:5]}"
json.dump(dict(base=a.base, latent=a.latent, step=step, ternary=n_tern, fp_trained=len(new) - n_tern, format="top86 group 128, dense unrotated"),
          open(os.path.join(a.out, "bonsai_export.json"), "w"), indent=1)
print("EXPORT_DONE", a.out)
