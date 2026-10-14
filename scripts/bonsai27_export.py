#!/usr/bin/env python3
"""Write a bonsai27_qat.py checkpoint (latent.pt) as an ordinary dense model any runtime can serve.

Every ternary tensor becomes unfold(top86(latent)), exactly what the trainer served; the trained full-precision
tensors are copied as trained; every other tensor of the base files (MTP head, vision tower) is copied unchanged.
With --qdir (prism basis only) the ternary tensors are also written in the folded GGUF layout, one <gguf name>.pt
each, as bonsai_hadamard_fold.py --override-dir expects.

    bonsai27_export.py --base B --latent RUN/latent.pt --out RUN/export [--prism-gguf F --qdir RUN/q]
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from bonsai_tern_lib import EMBED, HEAD, Basis, top86

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--latent", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prism-gguf", default="")
ap.add_argument("--qdir", default="")
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
dev = a.device
ck = torch.load(a.latent, map_location="cpu")
basis = Basis(ck["basis"], dev, a.prism_gguf)
lat, fp, step = ck["latents"], ck["fp"], ck["step"]
tied = HEAD not in lat


def file_key(k):                                  # trainer (text-model) name -> name in the base files
    return k if k == HEAD else "model.language_model." + k[len("model."):]


def served(k):
    w = lat[k].to(dev)
    out = torch.empty(w.shape, dtype=torch.bfloat16)
    n = max(1, 96_000_000 // w.shape[1]) if k in (EMBED, HEAD) else w.shape[0]
    for r in range(0, w.shape[0], n):
        out[r:r + n] = basis.unfold(k, top86(w[r:r + n].float())).to(torch.bfloat16).cpu()
    return out


if a.qdir:
    assert ck["basis"] == "prism", "--qdir needs a prism-basis checkpoint"
    os.makedirs(a.qdir, exist_ok=True)
    for k in lat:
        w = lat[k].to(dev)
        n = max(1, 96_000_000 // w.shape[1]) if k in (EMBED, HEAD) else w.shape[0]
        q = torch.cat([top86(w[r:r + n].float()).to(torch.bfloat16).cpu() for r in range(0, w.shape[0], n)])
        torch.save(q, os.path.join(a.qdir, basis.names[k] + ".pt"))
    print(f"wrote {len(lat)} folded tensors to {a.qdir}")

new = {file_key(k): None for k in lat}            # filled lazily, one shard at a time
new.update({file_key(k): v for k, v in fp.items()})
os.makedirs(a.out, exist_ok=True)
used = set()
for fn in sorted(os.listdir(a.base)):
    src = os.path.join(a.base, fn)
    if not fn.endswith(".safetensors"):
        if os.path.isfile(src) and not fn.endswith((".log", ".txt.aria2")) and fn not in ("urls.txt", "SHA256SUMS.hf"):
            shutil.copy2(src, os.path.join(a.out, fn))
        continue
    out = {}
    with safe_open(src, "pt") as f:
        meta = f.metadata()
        for k in f.keys():
            t = f.get_tensor(k)
            if k in new:
                tk = k if k == HEAD else "model." + k[len("model.language_model."):]
                v = served(tk) if new[k] is None else new[k]
                assert v.shape == t.shape, (k, v.shape, t.shape)
                out[k] = v.to(t.dtype)
                used.add(k)
            else:
                out[k] = t
    save_file(out, os.path.join(a.out, fn), metadata=meta)
    print(f"{fn}: {len(out)} tensors, {sum(1 for k in out if k in new)} replaced", flush=True)
    del out
missing = sorted(set(new) - used)
assert not missing, f"not found in the base files: {missing[:5]}"
json.dump(dict(base=a.base, latent=a.latent, step=step, basis=ck["basis"], ternary=len(lat), fp_trained=len(fp), tied_head=tied,
               format="top86 group 128, dense, model basis"), open(os.path.join(a.out, "bonsai_export.json"), "w"), indent=1)
print("EXPORT_DONE", a.out)
