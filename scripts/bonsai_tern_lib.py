#!/usr/bin/env python3
"""Shared pieces of the Bonsai-format ternary trainer and exporter (bonsai27_qat.py, bonsai27_export.py).

A ternary tensor is held as a LATENT in the rotated ("folded") basis. Its served weight is
unfold(top86(latent)): per group of 128 along the input axis the 86 largest |latent| keep their sign and share one
scale (mean of the kept magnitudes, snapped to bf16), the other 42 are zero; unfold carries that back to the model's
own basis. fold/unfold are orthogonal, so the gradient of the latent is fold(gradient of the served weight).

Two bases:
  seeded   signs from a fixed generator per input width, Hadamard block 1024 (512 where the width needs it). No layout
           change. This is the basis of bonsai4b_qat.py / bonsai17_qat.py and of the community reproductions.
  prism    signs, block and tensor list read from a PrismML Bonsai 2 GGUF, tensors in the GGUF row/column layout
           (linear-attention value rows regrouped, ssm_out columns permuted). Latents are then exactly what
           bonsai_hadamard_fold.py --override-dir packs.
"""
import math

import torch

G, NZERO = 128, 42
QNAMES = dict(self_attn=("q_proj", "k_proj", "v_proj", "o_proj"), linear_attn=("in_proj_qkv", "in_proj_z", "out_proj"),
              mlp=("gate_proj", "up_proj", "down_proj"))
FP_TRAIN = ("in_proj_a.", "in_proj_b.", "conv1d.", "A_log", "dt_bias")
GG2HF = {"attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_z", "ssm_out": "linear_attn.out_proj",
         "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
         "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj"}
EMBED, HEAD = "model.embed_tokens.weight", "lm_head.weight"


def fwht(x, block):
    """Normalized Sylvester Walsh-Hadamard transform over the last axis in independent blocks (its own inverse)."""
    shape = x.shape
    x = x.reshape(-1, shape[-1] // block, block)
    h = 1
    while h < block:
        x = x.reshape(x.shape[0], x.shape[1], block // (2 * h), 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2)
        h *= 2
    return x.reshape(shape) / math.sqrt(block)


def top86(w):
    """w [rows, in] float32 latent -> served ternary weight in the same basis (scale snapped to bf16)."""
    o, i = w.shape
    wg = w.reshape(o, i // G, G)
    aw = wg.abs()
    mask = aw > aw.kthvalue(NZERO, dim=-1, keepdim=True).values
    s = (aw * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1)
    s = s.to(torch.bfloat16).float()
    return (s * torch.sign(wg) * mask).reshape(o, i)


def trits(w):
    o, i = w.shape
    wg = w.reshape(o, i // G, G)
    aw = wg.abs()
    return (torch.sign(wg) * (aw > aw.kthvalue(NZERO, dim=-1, keepdim=True).values)).reshape(o, i).to(torch.int8)


def sr16(x):
    """float32 -> float16 with stochastic rounding, so updates smaller than the float16 spacing survive on average."""
    _, e = torch.frexp(x.abs().clamp(min=6.2e-5))
    ulp = torch.ldexp(torch.ones_like(x), e - 11)
    return (x + (torch.rand_like(x) - 0.5) * ulp).to(torch.float16)


class Basis:
    """fold / unfold for every ternary tensor of one model. Keys are HF state-dict names of the text model."""

    def __init__(self, mode, dev, prism_gguf=""):
        self.mode, self.dev, self.signs = mode, dev, {}
        self.names = {}                          # HF key -> GGUF tensor name (prism mode only)
        if mode == "prism":
            from bonsai_hadamard_fold import read_kv
            c = read_kv(prism_gguf, "prism.hadamard.")
            ak = read_kv(prism_gguf, "qwen35.ssm.")
            assert c["prism.hadamard.block_size"] == 1024, c["prism.hadamard.block_size"]
            off = 0
            for w in c["prism.hadamard.sign_widths"]:
                self.signs[w] = torch.tensor(c["prism.hadamard.sign_values"][off:off + w], dtype=torch.float32, device=dev)
                off += w
            self.NV, self.NK, width = ak["qwen35.ssm.time_step_rank"], ak["qwen35.ssm.group_count"], ak["qwen35.ssm.inner_size"]
            self.HDV, self.RPK = width // self.NV, self.NV // self.NK
            gi = torch.arange(width, device=dev)
            self.perm = gi % self.HDV + self.HDV * (((gi // self.HDV) // self.RPK) + self.NK * ((gi // self.HDV) % self.RPK))
            self.inv_perm = torch.argsort(self.perm)
            for nm in list(c["prism.hadamard.weight_names"]) + list(c["prism.hadamard.inverse_weight_names"]):
                self.names[self.hf_key(nm)] = nm
        else:
            assert mode == "seeded", mode

    @staticmethod
    def hf_key(nm):
        if nm == "token_embd.weight":
            return EMBED
        if nm == "output.weight":
            return HEAD
        _, i, kind, _ = nm.split(".")
        return f"model.layers.{i}.{GG2HF[kind]}.weight"

    def block(self, n):
        if self.mode == "prism":
            return 1024
        assert n % 512 == 0, n
        return 1024 if n % 1024 == 0 else 512

    def sign(self, n):
        if n not in self.signs:
            assert self.mode == "seeded", f"no signs for width {n} in the reference file"
            g = torch.Generator().manual_seed(1234 + n)
            self.signs[n] = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(self.dev)
        return self.signs[n]

    # --- GGUF layout of the linear-attention tensors (prism mode) ---
    def _re(self, t, dim, tiled_to_grouped):
        shape = list(t.shape)
        a, b = (self.RPK, self.NK) if tiled_to_grouped else (self.NK, self.RPK)
        ns = shape[:dim] + [a, b, self.HDV] + shape[dim + 1:]
        pr = list(range(len(ns)))
        pr[dim], pr[dim + 1] = pr[dim + 1], pr[dim]
        return t.reshape(*ns).permute(*pr).contiguous().reshape(*shape)

    def _layout(self, key, t, to_gguf):
        if self.mode != "prism":
            return t
        f = (lambda u, d: self._re(u, d, False)) if to_gguf else (lambda u, d: self._re(u, d, True))
        if ".in_proj_qkv." in key:
            qk = t.shape[0] - self.NV * self.HDV
            return torch.cat([t[:qk], f(t[qk:], 0)], 0)
        if ".in_proj_z." in key:
            return f(t, 0)
        if ".linear_attn.out_proj." in key:
            return f(t, 1)
        return t

    def fold(self, key, w):
        """Model-basis float32 weight (or its gradient) -> latent basis. Row subsets are fine for the embedding / head."""
        n = w.shape[1]
        g = self._layout(key, w, True)
        if self.mode == "prism" and ".linear_attn.out_proj." in key:
            g = g[:, self.perm]
        return fwht(g * self.sign(n), self.block(n))

    def unfold(self, key, q):
        n = q.shape[1]
        u = fwht(q, self.block(n)) * self.sign(n)
        if self.mode == "prism" and ".linear_attn.out_proj." in key:
            u = u[:, self.inv_perm]
        return self._layout(key, u, False)


def ternary_keys(model, basis):
    """HF keys of every tensor that is ternary, in model order, checked against the reference file in prism mode."""
    keys = [EMBED]
    for i, layer in enumerate(model.model.layers):
        for par, ns in QNAMES.items():
            if getattr(layer, par, None) is not None:
                keys += [f"model.layers.{i}.{par}.{n}.weight" for n in ns]
    if not model.config.tie_word_embeddings:
        keys.append(HEAD)
    if basis.mode == "prism":
        ref = set(basis.names)
        assert set(keys) == ref, f"ternary set differs from the reference file: {sorted(set(keys) ^ ref)[:6]}"
    return keys
