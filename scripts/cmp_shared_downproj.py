#!/usr/bin/env python3
"""cmp_shared_downproj.py — verify shared-alpha (=1.2) on the always-on dense FFN
survived the DERN routed-expert fold.

Compares model.language_model.layers.{L}.mlp.down_proj.weight (the SHARED dense FFN
output projection that router_shared_upweight.py scales by alpha) between the published
v7-coder student (shared-alpha baked in) and the dern11 redistribution build.

If redist.py only rewrote the routed-expert stack (experts.down_proj/gate_up_proj),
the shared mlp.down_proj.weight must be BIT-IDENTICAL across the two models -> alpha intact.
A nonzero diff would mean the fold disturbed the shared FFN (alpha diluted/lost).
"""
import json
import os
import torch
from safetensors import safe_open

STUDENT = "/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-it"
DERN11  = "/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-dern11-it"
LAYERS  = [0, 15, 29]   # spot-check shallow / mid / deep
KEY_T   = "model.language_model.layers.{L}.mlp.down_proj.weight"   # SHARED dense FFN
KEY_E   = "model.language_model.layers.{L}.experts.down_proj"      # ROUTED experts (should DIFFER)


def open_index(d):
    """Return (weight_map dict or None, single_file path or None)."""
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return json.load(f)["weight_map"], None
    single = os.path.join(d, "model.safetensors")
    if os.path.exists(single):
        return None, single
    raise FileNotFoundError(f"no safetensors in {d}")


_handles = {}
def get_tensor(d, key):
    wmap, single = open_index(d)
    if single is not None:
        path = single
    else:
        if key not in wmap:
            return None
        path = os.path.join(d, wmap[key])
    if path not in _handles:
        _handles[path] = safe_open(path, framework="pt", device="cpu")
    h = _handles[path]
    if key not in h.keys():
        return None
    return h.get_tensor(key)


def cmp(tag, key_tmpl):
    print(f"\n=== {tag} ===")
    for L in LAYERS:
        k = key_tmpl.format(L=L)
        a = get_tensor(STUDENT, k)
        b = get_tensor(DERN11, k)
        if a is None or b is None:
            print(f"  L{L:<2} key MISSING  student={a is not None} dern11={b is not None}  ({k})")
            continue
        a32, b32 = a.float(), b.float()
        na, nb = a32.norm().item(), b32.norm().item()
        if a.shape != b.shape:
            print(f"  L{L:<2} SHAPE DIFF student={tuple(a.shape)} dern11={tuple(b.shape)}")
            continue
        maxabs = (a32 - b32).abs().max().item()
        ident = torch.equal(a, b)
        print(f"  L{L:<2} student_norm={na:.4f} dern11_norm={nb:.4f} "
              f"max|diff|={maxabs:.3e} bit_identical={ident}")


print(f"student: {STUDENT}")
print(f"dern11 : {DERN11}")
cmp("SHARED dense FFN mlp.down_proj.weight (alpha target — expect IDENTICAL)", KEY_T)
cmp("ROUTED experts.down_proj (redist target — expect DIFFERENT)", KEY_E)
