#!/usr/bin/env python
"""attn_backend_test.py — which attention backend serves Gemma 4's head_dim=256
memory-efficiently? FA2 2.8.3 (conda-forge sm_120) rejects the hdim-256 forward
kernel on Blackwell, so we need a fallback that does NOT materialise the S*S
score matrix at long context. Tests flash_attn_func (fwd+bwd) at hd in {128,192,
256}, then each SDPA backend at hd=256/S=16384 with causal and with an additive
mask (Gemma 4 passes an additive bias on full-attn layers), reporting peak VRAM.
A backend is viable iff it succeeds AND peak << the ~2GB an S*S bf16 matrix would
cost at S=16384 (16384^2 * 2B = 0.5GB fwd, but it grows N^2 -> at 256k it is 64GB).
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

dev = "cuda"


def fa_test(hd):
    try:
        from flash_attn import flash_attn_func
        q = torch.randn(1, 4096, 8, hd, device=dev, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, 4096, 8, hd, device=dev, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, 4096, 8, hd, device=dev, dtype=torch.bfloat16, requires_grad=True)
        o = flash_attn_func(q, k, v, causal=True)
        o.sum().backward()
        return "OK"
    except Exception as e:
        return type(e).__name__ + ": " + str(e)[:80]


def sdpa_test(backend, mask):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    S, H, hd = 16384, 8, 256
    try:
        q = torch.randn(1, H, S, hd, device=dev, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, H, S, hd, device=dev, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, H, S, hd, device=dev, dtype=torch.bfloat16, requires_grad=True)
        with sdpa_kernel(backend):
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None))
        o.sum().backward()
        return "OK peak={:.2f}GB".format(torch.cuda.max_memory_allocated() / 1e9)
    except Exception as e:
        return type(e).__name__ + ": " + str(e)[:70]


print("flash_attn version:", __import__("flash_attn").__version__)
for hd in (128, 192, 256):
    print("  flash_attn_func hd={}: {}".format(hd, fa_test(hd)))

print("--- SDPA backends @ hd=256, S=16384 ---")
S = 16384
am = torch.zeros(1, 1, S, S, device=dev, dtype=torch.bfloat16)
cases = [
    ("FLASH      causal   ", SDPBackend.FLASH_ATTENTION, None),
    ("EFFICIENT  causal   ", SDPBackend.EFFICIENT_ATTENTION, None),
    ("CUDNN      causal   ", SDPBackend.CUDNN_ATTENTION, None),
    ("EFFICIENT  +addMask ", SDPBackend.EFFICIENT_ATTENTION, am),
    ("CUDNN      +addMask ", SDPBackend.CUDNN_ATTENTION, am),
]
for name, be, mask in cases:
    print("  {}: {}".format(name, sdpa_test(be, mask)))
