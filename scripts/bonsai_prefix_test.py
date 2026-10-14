#!/usr/bin/env python3
"""bonsai_prefix_test.py — was Bonsai 2 fitted layer-by-layer in order, or jointly?

Sequential compensation predicts: layer L's ternary weights are tuned for the inputs the ALREADY
QUANTIZED layers < L produce, so PrismML's layer L reproduces the base layer's output better on
PrismML's own upstream inputs than on base inputs. Joint training predicts no such preference.

Builds two copies of the first L+1 blocks of Qwen3.8-27B: base, and one whose every tensor is
replaced by PrismML's (ternary ones dequantized and UNFOLDED back to the primal basis; the rest
copied). Captures the input x of blk.L <module> in both, then compares, for that module's weight
W (base) and P (PrismML, primal basis):
   A  |P x_base  - W x_base|      the single-layer number already measured
   B  |P x_prism - W x_base|      what sequential compensation would make SMALL
   C  |W x_prism - W x_base|      upstream drift alone
Also reports, as forensics, which NON-ternary tensors PrismML changed relative to base.
"""
import argparse
import json

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForCausalLM

from bonsai_hadamard_fold import fwht_blocks, read_kv
from bonsai_layer_lab import decode_pq2, tensor_bytes

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--base-gguf", required=True)
ap.add_argument("--prism-gguf", required=True)
ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--module", default="self_attn.k_proj")
ap.add_argument("--gguf-module", default="attn_k")
ap.add_argument("--text", required=True)
ap.add_argument("--n-tokens", type=int, default=8192)
ap.add_argument("--out", required=True)
ap.add_argument("--save-pack", help="save PrismML trits/scales + layout maps + the PrismML-prefix state dict for the QAT pilot")
ap.add_argument("--device", default="cuda:1")
a = ap.parse_args()
dev = a.device
L = a.layer

c = read_kv(a.prism_gguf, "prism.hadamard.")
arch_kv = read_kv(a.prism_gguf, "qwen35.ssm.")
signs, off = {}, 0
for w in c["prism.hadamard.sign_widths"]:
    signs[w] = torch.tensor(c["prism.hadamard.sign_values"][off:off + w], dtype=torch.float32, device=dev)
    off += w
n_v, n_k, width = arch_kv["qwen35.ssm.time_step_rank"], arch_kv["qwen35.ssm.group_count"], arch_kv["qwen35.ssm.inner_size"]
hd, rep = width // n_v, n_v // n_k
gi = torch.arange(width, device=dev)
ii, rk = gi % hd, gi // hd
perm = ii + hd * ((rk // rep) + n_k * (rk % rep))
folded = set(c["prism.hadamard.weight_names"])


def gguf_tensor(path, name):
    """float32 tensor in numpy [out, in] order, PRIMAL basis (PrismML ternary ones are unfolded)."""
    typ, dims, raw = tensor_bytes(path, name)
    shp = tuple(reversed([int(d) for d in dims]))
    if typ == 0:
        return torch.from_numpy(np.frombuffer(raw, np.float32).copy()).reshape(shp).to(dev), "F32"
    if typ == 30:
        u = torch.from_numpy(np.frombuffer(raw, np.uint16).astype(np.int32)).to(dev)
        return (u << 16).view(torch.float32).reshape(shp), "BF16"
    if typ == 142:
        T, d = decode_pq2(raw, int(np.prod(shp)))
        P = (torch.from_numpy(T).to(dev).float() * torch.from_numpy(d).to(dev)[:, None]).reshape(shp)
        s = signs[shp[-1]]
        if name == "token_embd.weight":
            return s * fwht_blocks(P, 1024), "PQ2_0"            # latent rows: h = s * H(z)
        assert name in folded, name
        U = fwht_blocks(P, 1024) * s                             # = perm_cols(W)
        if ".ssm_out." in name:
            Wp = torch.empty_like(U)
            Wp[:, perm] = U
            U = Wp
        return U, "PQ2_0"
    raise SystemExit(f"unhandled type {typ} for {name}")


cfg = json.load(open(f"{a.model_dir}/config.json"))
tc = cfg["text_config"]
tc["num_hidden_layers"] = L + 1
for k, v in list(tc.items()):
    if isinstance(v, list) and len(v) == 64:
        tc[k] = v[:L + 1]
config = Qwen3_5Config(**cfg).text_config
idx = json.load(open(f"{a.model_dir}/model.safetensors.index.json"))["weight_map"]


def build():
    m = Qwen3_5ForCausalLM(config).to(torch.bfloat16)
    want, sd = m.state_dict(), {}
    for k, shard in idx.items():
        cand = k.replace("model.language_model.", "model.")
        if cand in want:
            with safe_open(f"{a.model_dir}/{shard}", "pt") as f:
                sd[cand] = f.get_tensor(k)
    miss = [k for k in want if k not in sd and "lm_head" not in k and not k.startswith("model.norm")]
    assert not miss, miss[:4]
    m.load_state_dict(sd, strict=False)
    return m.float().to(dev).eval()


base = build()
prism = build()
bsd = dict(base.state_dict())

# ---- map every GGUF tensor of blocks 0..L (+ token_embd) onto an HF parameter BY VALUE on the base pair ----
names = ["token_embd.weight"]
import subprocess  # noqa: E402  (tensor listing reuses the KV-only reader)
lst = subprocess.run(["/root/anaconda3/envs/omnimergekit/bin/python", "/srv/ml/scripts/gguf_read_prism_meta.py",
                      a.prism_gguf, "--tensors"], capture_output=True, text=True).stdout.splitlines()[1:]
for ln in lst:
    nm = ln.split()[0]
    if nm.startswith("blk.") and int(nm.split(".")[1]) <= L:
        names.append(nm)

report, changed, unmapped, n_set = {}, {}, [], 0
pack = {"tern": {}, "fp_keys": {}}
psd = dict(prism.state_dict())
NK, NV, HDV = int(n_k), int(n_v), int(hd)
RPK = NV // NK


def reorder_v(t, dim, head_dim):
    """convert_hf_to_gguf._reorder_v_heads: grouped (by K head) -> tiled, along `dim`."""
    shape = list(t.shape)
    ns = shape[:dim] + [NK, RPK, head_dim] + shape[dim + 1:]
    pr = list(range(len(ns)))
    pr[dim], pr[dim + 1] = pr[dim + 1], pr[dim]
    return t.reshape(*ns).permute(*pr).contiguous().reshape(*shape)


def to_gguf(key, t):
    """The converter's HF -> GGUF transform for this parameter (layout only + the two value maps)."""
    val = "id"
    if "linear_attn." in key:
        if ".in_proj_qkv." in key:
            qk = t.shape[0] - NV * HDV
            t = torch.cat([t[:qk], reorder_v(t[qk:], 0, HDV)], 0)
        elif ".in_proj_z." in key:
            t = reorder_v(t, 0, HDV)
        elif ".in_proj_a." in key or ".in_proj_b." in key:
            t = reorder_v(t, 0, 1)
        elif key.endswith(".A_log") or key.endswith(".dt_bias"):
            t = reorder_v(t.unsqueeze(-1), 0, 1).squeeze(-1)
        elif ".conv1d" in key:
            d = t.squeeze()
            qk = d.shape[0] - NV * HDV
            t = torch.cat([d[:qk], reorder_v(d[qk:], 0, HDV)], 0)
        elif ".out_proj." in key:
            t = reorder_v(t, 1, HDV)
    if key.endswith(".A_log"):
        val = "neg_exp"
    elif key.endswith("norm.weight") and not key.endswith("linear_attn.norm.weight"):
        val = "plus1"
    return t, val


def fwd_val(t, val):
    return -torch.exp(t) if val == "neg_exp" else (t + 1 if val == "plus1" else t)


def inv_val(g, val):
    return torch.log(-g) if val == "neg_exp" else (g - 1 if val == "plus1" else g)


for nm in names:
    gb, _ = gguf_tensor(a.base_gguf, nm)
    layer = None if nm == "token_embd.weight" else int(nm.split(".")[1])
    cands = [k for k in bsd if bsd[k].numel() == gb.numel() and
             (("embed_tokens" in k) if layer is None else (f"layers.{layer}." in k))]
    hit = None
    for k in cands:
        tb, val = to_gguf(k, bsd[k].float())
        got = fwd_val(tb, val).reshape(-1)
        # the base GGUF went bf16 -> (transform) -> F32/BF16; allow the value maps' float rounding
        if torch.allclose(got, gb.reshape(-1).to(got.device), rtol=2e-3, atol=1e-6) and \
           (val != "id" or torch.equal(got, gb.reshape(-1).to(got.device))):
            hit = (k, val)
            break
    gp, ptyp = gguf_tensor(a.prism_gguf, nm)
    fam = nm if layer is None else nm.split(".", 2)[2]
    rel = float((gp - gb).norm() / gb.norm().clamp(min=1e-30))
    changed.setdefault(f"{fam} [{ptyp}]", []).append(rel)
    if hit is None:
        unmapped.append(nm)
        continue
    k, val = hit
    # invert the layout by pushing an index tensor through the same transform
    ix, _ = to_gguf(k, torch.arange(bsd[k].numel(), device=dev).reshape(bsd[k].shape))
    flat = torch.empty(bsd[k].numel(), device=dev, dtype=torch.float32)
    flat[ix.reshape(-1)] = inv_val(gp.reshape(-1).float(), val)
    psd[k].copy_(flat.reshape(psd[k].shape).to(psd[k].dtype))
    if ptyp == "PQ2_0" and nm != "token_embd.weight":
        _t, _dims, _raw = tensor_bytes(a.prism_gguf, nm)
        Tq, dq = decode_pq2(_raw, gp.numel())
        pack["tern"][nm] = {"key": k, "ix": ix.reshape(-1).cpu(), "gshape": tuple(gp.shape),
                            "T": torch.from_numpy(Tq.copy()), "d": torch.from_numpy(dq.copy())}
    elif nm != "token_embd.weight":
        pack["fp_keys"][nm] = k
    n_set += 1
report["tensors_replaced_with_prism"] = n_set
report["unmapped_gguf_tensors"] = unmapped
report["prism_vs_base_rel_change_by_family (mean over blocks)"] = {k: float(np.mean(v)) for k, v in sorted(changed.items())}
print(json.dumps(report, indent=1), flush=True)
if unmapped:
    raise SystemExit("FATAL: some GGUF tensors have no value-identical HF parameter; the prefix would be a hybrid")
prism.load_state_dict(psd, strict=False)
if a.save_pack:
    pack["prism_sd"] = {k: v.cpu() for k, v in psd.items()}
    torch.save(pack, a.save_pack)
    print("pack saved", a.save_pack, flush=True)

tok = AutoTokenizer.from_pretrained(a.model_dir)
ids = tok(open(a.text).read()[:400000], return_tensors="pt").input_ids[0][: a.n_tokens]


def capture(m):
    got = []
    h = m.model.layers[L].get_submodule(a.module).register_forward_pre_hook(
        lambda mod, inp: got.append(inp[0].detach().float().reshape(-1, inp[0].shape[-1])))
    with torch.no_grad():
        for i in range(0, len(ids), 1024):
            m.model(input_ids=ids[i:i + 1024][None].to(dev))
    h.remove()
    return torch.cat(got)


xb, xp = capture(base), capture(prism)
gname = f"blk.{L}.{a.gguf_module}.weight"
Wb, _ = gguf_tensor(a.base_gguf, gname)
P, _ = gguf_tensor(a.prism_gguf, gname)
y0 = xb @ Wb.T
rel = lambda y: float((y - y0).norm() / y0.norm())  # noqa: E731
res = {
    "target": gname,
    "input drift |x_prism - x_base| / |x_base|": float((xp - xb).norm() / xb.norm()),
    "input cosine (mean per token)": float(torch.nn.functional.cosine_similarity(xp, xb, dim=1).mean()),
    "A  |P x_base  - W x_base|": rel(xb @ P.T),
    "B  |P x_prism - W x_base|": rel(xp @ P.T),
    "C  |W x_prism - W x_base|": rel(xp @ Wb.T),
    "D  |P x_prism - W x_prism| / |W x_prism|": float((xp @ P.T - xp @ Wb.T).norm() / (xp @ Wb.T).norm()),
}
report["test"] = res
print(json.dumps(res, indent=1), flush=True)
json.dump(report, open(a.out, "w"), indent=1)
print("PREFIX_DONE")
