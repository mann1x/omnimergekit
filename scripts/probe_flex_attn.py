#!/usr/bin/env python
"""probe_flex_attn.py - does torch FlexAttention (Triton) serve hd=512 on sm_120?
The SDPA C++ backends (FLASH/EFFICIENT/CUDNN) have NO hd=512 kernel on sm_120, but
FlexAttention compiles a Triton kernel with arbitrary head_dim + block-sparse (linear)
memory. If it works at hd=512 causal, the Gemma4 26B-A4B 5 global layers (hd=512) can
run a single full forward over 87k tokens -- no chunked prefill needed. Tests causal at
hd in {256,512}, GQA(16/8), growing S; also a sliding-window mask at hd=256. Reports
peak VRAM (linear => block kernel works; OOM/error => no kernel)."""
import torch
import torch.nn.functional as F  # noqa: F401
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

dev = "cuda"
HQ, HKV = 16, 8
WINDOW = 1024
flex = torch.compile(flex_attention)


def mk(h, S, HD):
    return torch.randn(1, h, S, HD, device=dev, dtype=torch.bfloat16)


def causal(b, h, q, kv):
    return q >= kv


def sliding(b, h, q, kv):
    return (q >= kv) & (q - kv < WINDOW)


def trial(tag, S, HD, mod):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        q, k, v = mk(HQ, S, HD), mk(HKV, S, HD), mk(HKV, S, HD)
        bm = create_block_mask(mod, B=None, H=None, Q_LEN=S, KV_LEN=S, device=dev)
        with torch.no_grad():
            o = flex(q, k, v, block_mask=bm, enable_gqa=True)
        torch.cuda.synchronize()
        pk = torch.cuda.max_memory_allocated() / 1e9
        print("  %-22s hd=%3d S=%6d: OK   peak=%.2fGB out=%s" % (tag, HD, S, pk, tuple(o.shape)), flush=True)
        del q, k, v, o, bm
    except Exception as e:  # noqa: BLE001
        print("  %-22s hd=%3d S=%6d: %s: %s" % (tag, HD, S, type(e).__name__, str(e)[:90]), flush=True)


print("torch", torch.__version__, "cap", torch.cuda.get_device_capability(0), flush=True)
for HD in (256, 512):
    print("=== causal head_dim=%d ===" % HD, flush=True)
    for S in (32768, 65536):
        trial("FLEX causal", S, HD, causal)
print("=== sliding(win=%d) head_dim=256 ===" % WINDOW, flush=True)
for S in (32768, 65536):
    trial("FLEX sliding", S, 256, sliding)
print("PROBE_FLEX_DONE", flush=True)
