#!/usr/bin/env python3
"""bonsai_pt2_full.py — whole-model PT2-style ternarization of Qwen3.8-27B into the Bonsai 2 format.

Same quantizer as bonsai_pt2_blocks.py (ITF + GPTQ + AGA, folded basis, one scale per 128-group,
no offset, no column reordering), over all 64 blocks + token_embd + output head. Pure PTQ.

The model sits on one GPU in bf16. Blocks are processed in order; each block's Hessians come from
calibration text run through the ALREADY-TERNARIZED prefix (early exit after the block). Every
quantized tensor is written to --qdir in the folded GGUF layout as exact +-d bf16 values, which
`bonsai_hadamard_fold.py --override-dir` then drops into the GGUF for the codec to pack losslessly.
RESUMABLE: a block whose tensors are all in --qdir is loaded, not recomputed.
"""
import argparse
import json
import os
import time

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForCausalLM

from bonsai_hadamard_fold import fwht_blocks, read_kv

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--prism-gguf", required=True, help="contract source (signs, weight names)")
ap.add_argument("--calib-text", required=True)
ap.add_argument("--qdir", required=True)
ap.add_argument("--n-calib", type=int, default=96)
ap.add_argument("--ctx", type=int, default=1024)
ap.add_argument("--damp", type=float, default=0.01)
ap.add_argument("--max-blocks", type=int, default=64)
ap.add_argument("--device", default="cuda:1")
a = ap.parse_args()
dev, G = a.device, 128
os.makedirs(a.qdir, exist_ok=True)

c = read_kv(a.prism_gguf, "prism.hadamard.")
ak = read_kv(a.prism_gguf, "qwen35.ssm.")
signs, off = {}, 0
for w in c["prism.hadamard.sign_widths"]:
    signs[w] = torch.tensor(c["prism.hadamard.sign_values"][off:off + w], dtype=torch.float32, device=dev)
    off += w
NV, NK, width = ak["qwen35.ssm.time_step_rank"], ak["qwen35.ssm.group_count"], ak["qwen35.ssm.inner_size"]
HDV, RPK = width // NV, NV // NK
gi = torch.arange(width, device=dev)
perm = gi % HDV + HDV * (((gi // HDV) // RPK) + NK * ((gi // HDV) % RPK))
inv_perm = torch.argsort(perm)
contract = list(c["prism.hadamard.weight_names"]) + list(c["prism.hadamard.inverse_weight_names"])

HF = {"attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_z", "ssm_out": "linear_attn.out_proj",
      "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
      "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj"}


def hf_key(nm):
    if nm == "token_embd.weight":
        return "model.embed_tokens.weight"
    if nm == "output.weight":
        return "lm_head.weight"
    _, i, kind, _ = nm.split(".")
    return f"model.layers.{i}.{HF[kind]}.weight"


def reorder_v(t, dim, head_dim):
    shape = list(t.shape)
    ns = shape[:dim] + [NK, RPK, head_dim] + shape[dim + 1:]
    pr = list(range(len(ns)))
    pr[dim], pr[dim + 1] = pr[dim + 1], pr[dim]
    return t.reshape(*ns).permute(*pr).contiguous().reshape(*shape)


def unreorder_v(t, dim, head_dim):
    """inverse of reorder_v: tiled -> grouped"""
    shape = list(t.shape)
    ns = shape[:dim] + [RPK, NK, head_dim] + shape[dim + 1:]
    pr = list(range(len(ns)))
    pr[dim], pr[dim + 1] = pr[dim + 1], pr[dim]
    return t.reshape(*ns).permute(*pr).contiguous().reshape(*shape)


def from_gguf_layout(key, t):
    if ".in_proj_qkv." in key:
        qk = t.shape[0] - NV * HDV
        return torch.cat([t[:qk], unreorder_v(t[qk:], 0, HDV)], 0)
    if ".in_proj_z." in key:
        return unreorder_v(t, 0, HDV)
    if ".linear_attn.out_proj." in key:
        return unreorder_v(t, 1, HDV)
    return t


def to_gguf_layout(key, t):
    if ".in_proj_qkv." in key:
        qk = t.shape[0] - NV * HDV
        return torch.cat([t[:qk], reorder_v(t[qk:], 0, HDV)], 0)
    if ".in_proj_z." in key:
        return reorder_v(t, 0, HDV)
    if ".linear_attn.out_proj." in key:
        return reorder_v(t, 1, HDV)
    return t


cfg = json.load(open(f"{a.model_dir}/config.json"))
config = Qwen3_5Config(**cfg).text_config
idx = json.load(open(f"{a.model_dir}/model.safetensors.index.json"))["weight_map"]
t0 = time.time()
model = Qwen3_5ForCausalLM(config).to(torch.bfloat16)
want, sd0, opened = model.state_dict(), {}, {}
for k, shard in idx.items():
    cand = k.replace("model.language_model.", "model.")
    if cand in want:
        if shard not in opened:
            opened[shard] = safe_open(f"{a.model_dir}/{shard}", "pt")
        sd0[cand] = opened[shard].get_tensor(k)
miss = [k for k in want if k not in sd0]
print(f"loaded {len(sd0)} tensors in {time.time() - t0:.0f}s; missing: {miss[:6]}", flush=True)
if any("lm_head" not in k for k in miss):
    raise SystemExit("FATAL: base weights missing")
if "lm_head.weight" in miss:
    raise SystemExit("FATAL: lm_head not in the checkpoint (tied?) — the output head path needs handling")
model.load_state_dict(sd0, strict=True)
del sd0
model = model.to(dev).eval()
sd = model.state_dict()
print(f"model on {dev}: {torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB", flush=True)

spec = {}
for nm in contract:
    k = hf_key(nm)
    assert k in sd, (nm, k)
    out_f, in_f = sd[k].shape
    # the layout is applied on demand: a stored index map for all 402 tensors would be 27B int64 (bug-758)
    colmap = torch.arange(in_f, device=dev)
    is_out = ".ssm_out." in nm
    if is_out:
        colmap = reorder_v(colmap[None], 1, HDV)[0]
    spec[nm] = {"key": k, "shape": (out_f, in_f), "is_out": is_out,
                "xmap": colmap[perm] if is_out else colmap, "s": signs[in_f]}


def to_folded(nm):
    e = spec[nm]
    g = to_gguf_layout(e["key"], sd[e["key"]].float())
    if e["is_out"]:
        g = g[:, perm]
    return fwht_blocks(g * e["s"], 1024)


def apply_folded(nm, q):
    e = spec[nm]
    u = fwht_blocks(q.float(), 1024) * e["s"]
    if e["is_out"]:
        u = u[:, inv_perm]
    sd[e["key"]].copy_(from_gguf_layout(e["key"], u).to(torch.bfloat16))


def snap_bf16(x):
    return x.float().to(torch.bfloat16).to(x.dtype)


def itf(w, iters=10):
    aw = w.abs()
    m = aw > 0.75 * aw.mean(1, keepdim=True)
    al = (aw * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
    for _ in range(iters):
        T = torch.round(w / al.clamp(min=1e-12)).clamp(-1, 1)
        al = (w * T).sum(1, keepdim=True) / (T * T).sum(1, keepdim=True).clamp(min=1)
    return al.clamp(min=1e-12)


def quantize(Wf32, H, tm=None):
    """Wf32 [out, in] folded float32 (GPU). H [in, in] float64 or None (ITF only). Returns bf16-exact T*d."""
    n_out, n_in = Wf32.shape
    ng = n_in // G
    out = torch.empty(n_out, n_in, device=dev, dtype=torch.bfloat16)
    if H is not None:
        Hd = H + a.damp * H.diag().mean() * torch.eye(n_in, device=dev, dtype=H.dtype)
        Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True)
        del Hd
        eye = torch.eye(ng, device=dev, dtype=torch.float64)
    rows = max(256, int(6.0e9 // (n_in * 8 * 4)))
    for r0 in range(0, n_out, rows):
        t_q = time.time()
        Wf = Wf32[r0:r0 + rows].double()
        W = Wf.clone()
        Q = torch.zeros_like(W)
        for i1 in range(0, n_in, G):
            i2 = i1 + G
            W1 = W[:, i1:i2].clone()
            al = itf(W1)[:, 0]
            if H is None:
                Q[:, i1:i2] = torch.round(W1 / al[:, None]).clamp(-1, 1) * al[:, None]
                continue
            Err = torch.zeros_like(W1)
            Hi = Hinv[i1:i2, i1:i2]
            for j in range(G):
                w = W1[:, j]
                q = torch.round(w / al).clamp(-1, 1) * al
                Q[:, i1 + j] = q
                err = (w - q) / Hi[j, j]
                W1[:, j:] -= err[:, None] * Hi[j, j:][None]
                Err[:, j] = err
            W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
        T = torch.sign(Q)
        scale = Q.abs().reshape(-1, ng, G).amax(2)
        if tm is not None:
            torch.cuda.synchronize(dev)
            tm["gptq"] += time.time() - t_q
            t_q = time.time()
        if H is not None:                                            # AGA: refit group scales, trits frozen
            # A[r,g,h] = T_rg H_gh T_rh^T. Only 128 of each Tg row are non-zero, so build it one group
            # at a time from P_g = T[:, g] @ H[g, :]  (cost rows*in*in in total, not rows*ng*in*in).
            nr = T.shape[0]
            Tv = T.view(nr, ng, G)
            A = torch.empty(nr, ng, ng, device=dev, dtype=torch.float64)
            for g in range(ng):
                P = T[:, g * G:(g + 1) * G] @ H[g * G:(g + 1) * G, :]
                A[:, g, :] = (P.view(nr, ng, G) * Tv).sum(-1)
            b = ((Wf @ H).view(nr, ng, G) * Tv).sum(-1)
            A = A + 1e-6 * A.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * eye
            new = torch.linalg.solve(A, b)
            del A, b, P
            scale = torch.where(new > 0, new, scale)
        if tm is not None:
            torch.cuda.synchronize(dev)
            tm["aga"] += time.time() - t_q
        scale = snap_bf16(scale)
        out[r0:r0 + rows] = (T * scale.repeat_interleave(G, 1)).to(torch.bfloat16)
        del Wf, W, Q, T
    return out


class Stop(Exception):
    pass


tok = AutoTokenizer.from_pretrained(a.model_dir)
cal = tok(open(a.calib_text).read()[:4_000_000], return_tensors="pt").input_ids[0]
st = torch.randint(0, len(cal) - a.ctx - 1, (a.n_calib,), generator=torch.Generator().manual_seed(0))
cal = torch.stack([cal[s_:s_ + a.ctx] for s_ in st])
print(f"calibration {cal.numel()} tokens", flush=True)


def qpath(nm):
    return os.path.join(a.qdir, nm + ".pt")


def run_group(names, stop_module):
    """Hessians for `names` from the current student, then quantize + apply + save."""
    todo = [n for n in names if not os.path.exists(qpath(n))]
    for n in names:
        if n not in todo:
            apply_folded(n, torch.load(qpath(n), map_location=dev))
    if not todo:
        return "loaded"
    tm = {"collect": 0.0, "gptq": 0.0, "aga": 0.0}
    t_c = time.time()
    Hs = {n: torch.zeros(spec[n]["shape"][1], spec[n]["shape"][1], device=dev, dtype=torch.float64) for n in todo}
    cnt = {n: 0 for n in todo}
    hooks = []
    for n in todo:
        def mk(n):
            e = spec[n]

            def hook(mod, inp):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])[:, e["xmap"]]
                xr = fwht_blocks(x * e["s"], 1024).double()
                Hs[n].add_(xr.T @ xr)
                cnt[n] += x.shape[0]
            return hook
        hooks.append(model.get_submodule(spec[n]["key"][: -len(".weight")]).register_forward_pre_hook(mk(n)))
    if stop_module is not None:
        def stop(mod, inp, out):
            raise Stop
        hooks.append(stop_module.register_forward_hook(stop))
    with torch.no_grad():
        for i in range(cal.shape[0]):
            try:
                model(input_ids=cal[i:i + 1].to(dev))
            except Stop:
                pass
    for h in hooks:
        h.remove()
    torch.cuda.synchronize(dev)
    tm["collect"] = time.time() - t_c
    for n in todo:
        Wf = to_folded(n)                      # from the ORIGINAL weights: sd still holds them for this group
        q = quantize(Wf, Hs[n] / max(cnt[n], 1), tm)
        torch.save(q.cpu(), qpath(n) + ".tmp")
        os.replace(qpath(n) + ".tmp", qpath(n))
        del Wf
        Hs[n] = None
        spec[n]["_q"] = q
    for n in todo:                             # apply only after the whole group is fitted
        apply_folded(n, spec[n].pop("_q"))
    torch.cuda.empty_cache()
    return "fitted  [collect %.0fs  gptq %.0fs  aga %.0fs]" % (tm["collect"], tm["gptq"], tm["aga"])


# token_embd: a lookup table — no input activations; weight-space ITF in the latent basis
nm = "token_embd.weight"
if os.path.exists(qpath(nm)):
    apply_folded(nm, torch.load(qpath(nm), map_location=dev))
else:
    q = quantize(to_folded(nm), None)
    torch.save(q.cpu(), qpath(nm) + ".tmp")
    os.replace(qpath(nm) + ".tmp", qpath(nm))
    apply_folded(nm, q)
    del q
print(f"[{time.time() - t0:6.0f}s] token_embd done", flush=True)

layers = model.model.layers
for b in range(min(a.max_blocks, len(layers))):
    names = [n for n in contract if n.startswith(f"blk.{b}.")]
    how = run_group(names, layers[b])
    print(f"[{time.time() - t0:6.0f}s] block {b:2d} {how} ({len(names)} tensors)", flush=True)
if a.max_blocks >= len(layers):
    how = run_group(["output.weight"], None)
    print(f"[{time.time() - t0:6.0f}s] output head {how}", flush=True)
    print("PT2_FULL_DONE", flush=True)
