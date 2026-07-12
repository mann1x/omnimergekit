#!/usr/bin/env python3
"""Definitive v6 vs v7 CD-Q4_K_M allocation diff — is the low tier IQ3_S (v7,
broken) or Q3_K (v6, healthy)?"""
from collections import Counter
from gguf import GGUFReader

FILES = {
    "v6 CD-Q4_K_M (healthy)": "/mnt/sdc/ml/v6_cd_compare/gemma-4-A4B-98e-v6-coder-it-CD-Q4_K_M.gguf",
    "v7 CD-Q4_K_M (ruminator)": "/mnt/sdc/ml/quant_sweep_gguf/gemma-4-A4B-98e-v7-coder-it-CD-Q4_K_M.gguf",
}


def types(p):
    return {t.name: t.tensor_type.name for t in GGUFReader(p).tensors}


for label, path in FILES.items():
    try:
        t = types(path)
    except Exception as e:
        print(f"{label}: LOAD FAILED {e}")
        continue
    h = dict(Counter(t.values()))
    low3 = {k: v for k, v in h.items() if k in ("IQ3_S", "Q3_K", "IQ3_XS", "IQ3_M", "Q3_K_S")}
    print(f"\n{label}")
    print(f"  histogram: {h}")
    print(f"  3-bit-class tensors: {low3}")
    print(f"  attn_q types: {dict(Counter(v for k, v in t.items() if 'attn_q' in k))}")
    print(f"  attn_output types: {dict(Counter(v for k, v in t.items() if 'attn_output' in k))}")
    print(f"  ffn_down types: {dict(Counter(v for k, v in t.items() if 'ffn_down' in k))}")
