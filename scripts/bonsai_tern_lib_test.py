#!/usr/bin/env python3
"""CPU self-test of bonsai_tern_lib: fold/unfold round trip and agreement with the bonsai_pt2_full.py convention.

    bonsai_tern_lib_test.py PRISM_GGUF
"""
import sys

import torch

from bonsai_hadamard_fold import fwht_blocks, read_kv
from bonsai_tern_lib import Basis, sr16, top86

b = Basis("prism", "cpu", sys.argv[1])
ak = read_kv(sys.argv[1], "qwen35.ssm.")
NV, NK, width = ak["qwen35.ssm.time_step_rank"], ak["qwen35.ssm.group_count"], ak["qwen35.ssm.inner_size"]
HDV, RPK = width // NV, NV // NK


def reorder_v(t, dim, head_dim):                  # verbatim from bonsai_pt2_full.py
    shape = list(t.shape)
    ns = shape[:dim] + [NK, RPK, head_dim] + shape[dim + 1:]
    pr = list(range(len(ns)))
    pr[dim], pr[dim + 1] = pr[dim + 1], pr[dim]
    return t.reshape(*ns).permute(*pr).contiguous().reshape(*shape)


def ref_fold(key, w):
    if ".in_proj_qkv." in key:
        qk = w.shape[0] - NV * HDV
        g = torch.cat([w[:qk], reorder_v(w[qk:], 0, HDV)], 0)
    elif ".in_proj_z." in key:
        g = reorder_v(w, 0, HDV)
    elif ".linear_attn.out_proj." in key:
        g = reorder_v(w, 1, HDV)[:, b.perm]
    else:
        g = w
    return fwht_blocks(g * b.signs[w.shape[1]], 1024)


torch.manual_seed(0)
print(f"NV {NV} NK {NK} width {width} HDV {HDV}; widths {sorted(b.signs)}; {len(b.names)} tensors in the contract")
for key, shape in (("model.layers.0.linear_attn.in_proj_qkv.weight", (10240, 5120)), ("model.layers.0.linear_attn.in_proj_z.weight", (6144, 5120)),
                   ("model.layers.0.linear_attn.out_proj.weight", (5120, 6144)), ("model.layers.3.self_attn.q_proj.weight", (12288, 5120)),
                   ("model.layers.0.mlp.down_proj.weight", (5120, 17408)), ("model.embed_tokens.weight", (4096, 5120))):
    w = torch.randn(shape)
    f = b.fold(key, w)
    rt = (b.unfold(key, f) - w).abs().max().item()
    ag = (f - ref_fold(key, w)).abs().max().item()
    gq, x = torch.randn(shape), torch.randn(shape)       # <unfold(x), g> == <x, fold(g)>: fold is the transpose of unfold
    adj = abs((b.unfold(key, x) * gq).sum().item() - (x * b.fold(key, gq)).sum().item()) / gq.numel()
    print(f"{key.split('.')[-2]:12s} round trip {rt:.2e}  vs pt2_full {ag:.2e}  adjoint {adj:.2e}")
    assert rt < 1e-4 and ag < 1e-5 and adj < 1e-5
q = top86(torch.randn(8, 1024))
print("top86 zeros per group:", sorted(set((q.reshape(8, 8, 128) == 0).sum(-1).flatten().tolist())))
x = torch.full((200000,), 0.0200003)
print(f"sr16 mean error {(sr16(x).float().mean() - 0.0200003).item():.2e} (plain half rounding error {(x.half().float().mean() - 0.0200003).item():.2e})")
print("LIB_TEST_OK")
