#!/usr/bin/env python
"""probe_attn_fwd.py - forward-only (no-grad) SDPA backend probe for Gemma 4 on sm_120.
Gemma 4 26B-A4B has TWO regimes: 25 sliding layers hd=256/window=1024, and 5 global
full-attn layers hd=512. Decides the long-context attention path for the agentic router
map. Tests FLASH/EFFICIENT/CUDNN causal at hd in {256,512}, GQA(16/8), growing S.
No flash_attn import. Reports peak VRAM (linear peak => block kernel; ~S^2 or OOM => math)."""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

dev = "cuda"
HQ, HKV = 16, 8


def mk(h, S, HD):
    return torch.randn(1, h, S, HD, device=dev, dtype=torch.bfloat16)


def trial(tag, S, HD, be):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        q, k, v = mk(HQ, S, HD), mk(HKV, S, HD), mk(HKV, S, HD)
        with torch.no_grad(), sdpa_kernel(be):
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        torch.cuda.synchronize()
        pk = torch.cuda.max_memory_allocated() / 1e9
        print("  %-26s hd=%3d S=%6d: OK   peak=%.2fGB" % (tag, HD, S, pk), flush=True)
        del q, k, v, o
    except Exception as e:  # noqa: BLE001
        print("  %-26s hd=%3d S=%6d: %s: %s" % (tag, HD, S, type(e).__name__, str(e)[:60]), flush=True)


print("torch", torch.__version__, "cap", torch.cuda.get_device_capability(0), flush=True)
for HD in (256, 512):
    print("=== head_dim=%d ===" % HD, flush=True)
    for S in (32768, 65536):
        trial("FLASH causal", S, HD, SDPBackend.FLASH_ATTENTION)
        trial("EFFICIENT causal", S, HD, SDPBackend.EFFICIENT_ATTENTION)
        trial("CUDNN causal", S, HD, SDPBackend.CUDNN_ATTENTION)
print("PROBE_DONE", flush=True)
