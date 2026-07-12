#!/usr/bin/env python
"""Characterize non-finite float tensors in a modelopt NVFP4A16 export dir.
Decide whether non-finite values are a real quant defect (weights) or an
expected artifact (scale/amax sentinels)."""
import sys, glob, os, collections
from safetensors import safe_open
import torch

d = sys.argv[1]
print("dir:", d)
files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
print("shards:", len(files))
tot = 0
bysuffix = collections.Counter()
badsuffix = collections.Counter()
examples = []
for f in files:
    with safe_open(f, framework="pt") as h:
        for k in h.keys():
            t = h.get_tensor(k); tot += 1
            suf = k.split(".")[-1]
            if t.is_floating_point():
                bysuffix[suf] += 1
                tf = t.float()  # isfinite is NotImplemented for Float8_e4m3fn (NVFP4 scales)
                if not torch.isfinite(tf).all():
                    nnan = int(torch.isnan(tf).sum())
                    ninf = int(torch.isinf(tf).sum())
                    badsuffix[suf] += 1
                    if len(examples) < 15:
                        examples.append((k, str(t.dtype), tuple(t.shape), nnan, ninf, int(t.numel())))
print("total tensors:", tot)
print("float tensors by suffix:", dict(bysuffix))
print("NON-FINITE float tensors by suffix:", dict(badsuffix))
print("examples (name, dtype, shape, #nan, #inf, numel):")
for e in examples:
    print("  ", e)
