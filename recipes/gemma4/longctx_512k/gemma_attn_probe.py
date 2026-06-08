#!/usr/bin/env python
"""gemma_attn_probe.py — nail down EXACTLY which attention call Gemma 4 emits at
training and which backend can serve it memory-efficiently at hd=256. Tests the
real-shape combinations: GQA (16 q / 8 kv heads), additive sliding-window mask vs
causal-no-mask, FA2 dense vs varlen vs softcap, and SDPA EFFICIENT/FLASH with
enable_gqa vs repeat_kv. Small S so it's fast; we only care which COMBINATIONS
dispatch, not their long-context memory (that's the model probe's job)."""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

dev = "cuda"
S, HQ, HKV, HD = 8192, 16, 8, 256
SW = 1024  # sliding window


def _ctx(be, fn):
    with sdpa_kernel(be):
        return fn()


def mk(h):
    return torch.randn(1, h, S, HD, device=dev, dtype=torch.bfloat16, requires_grad=True)


def band_mask():
    # additive sliding-window causal mask, dense (1,1,S,S) — what HF builds for sdpa
    i = torch.arange(S, device=dev)
    keep = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - SW)
    m = torch.zeros(S, S, device=dev, dtype=torch.bfloat16)
    m.masked_fill_(~keep, float("-inf"))
    return m[None, None]


def run(tag, fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        o = fn()
        o.sum().backward()
        print("  {:42s}: OK  peak={:.2f}GB".format(tag, torch.cuda.max_memory_allocated() / 1e9))
    except Exception as e:
        print("  {:42s}: {}: {}".format(tag, type(e).__name__, str(e)[:60]))


print("=== FA2 package kernels @ hd=256 ===")
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    q, k, v = mk(HKV), mk(HKV), mk(HKV)
    run("flash_attn_func dense causal", lambda: flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True))
    run("flash_attn_func dense +softcap=50", lambda: flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True, softcap=50.0))
    run("flash_attn_func dense +window(1024)", lambda: flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True, window_size=(SW, 0)))

    def varlen():
        qf = q.transpose(1, 2).reshape(S, HKV, HD)
        kf = k.transpose(1, 2).reshape(S, HKV, HD)
        vf = v.transpose(1, 2).reshape(S, HKV, HD)
        cu = torch.tensor([0, S], device=dev, dtype=torch.int32)
        return flash_attn_varlen_func(qf, kf, vf, cu, cu, S, S, causal=True)
    run("flash_attn_varlen_func causal", varlen)
except Exception as e:
    print("  FA2 import/setup failed:", e)

print("=== SDPA EFFICIENT/FLASH @ hd=256, GQA shapes ===")
qg, kg, vg = mk(HQ), mk(HKV), mk(HKV)
bm = band_mask()
for be, nm in [(SDPBackend.EFFICIENT_ATTENTION, "EFFICIENT"), (SDPBackend.FLASH_ATTENTION, "FLASH")]:
    # enable_gqa path (what HF uses when use_gqa_in_sdpa is True)
    run("{} enable_gqa causal".format(nm),
        lambda be=be: _ctx(be, lambda: F.scaled_dot_product_attention(qg, kg, vg, is_causal=True, enable_gqa=True)))
    run("{} enable_gqa +bandMask".format(nm),
        lambda be=be: _ctx(be, lambda: F.scaled_dot_product_attention(qg, kg, vg, attn_mask=bm, enable_gqa=True)))
    # repeat_kv path (matched heads, no enable_gqa)
    kr = kg.repeat_interleave(HQ // HKV, dim=1)
    vr = vg.repeat_interleave(HQ // HKV, dim=1)
    run("{} repeat_kv +bandMask".format(nm),
        lambda be=be, kr=kr, vr=vr: _ctx(be, lambda: F.scaled_dot_product_attention(qg, kr, vr, attn_mask=bm)))
