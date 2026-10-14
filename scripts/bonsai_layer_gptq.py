#!/usr/bin/env python3
"""bonsai_layer_gptq.py — can error-compensating PTQ reproduce PrismML's trits on one layer?

Finding that motivates this (bonsai_layer_lab.py, real activations, blk.3.attn_k): magnitude
top-86 matches 92.5% of PrismML's trits yet has 2.1x PrismML's output error (0.84 vs 0.39), and
a diagonal (per-column energy) weighting changes nothing. So the remaining 7.5% of trits are
chosen against the CORRELATED input structure — the signature of OBQ/GPTQ-style compensation
or of training. This runs GPTQ (Frantar et al., arXiv 2210.17323) in PrismML's folded basis with
ternary levels {-d, 0, +d} and one d per 128-weight group, and scores it two ways: output error
on held-out real activations, and trit agreement with PrismML's shipped layer.
"""
import argparse
import json

import torch
from safetensors import safe_open

from bonsai_hadamard_fold import fwht_blocks, read_kv
from bonsai_layer_lab import G, decode_pq2, tensor_bytes

ap = argparse.ArgumentParser()
ap.add_argument("--pq2", required=True)
ap.add_argument("--base-st", required=True)
ap.add_argument("--st-key", required=True)
ap.add_argument("--tensor", required=True)
ap.add_argument("--acts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:1")
a = ap.parse_args()
dev = a.device
torch.set_default_dtype(torch.float64)

typ, dims, raw = tensor_bytes(a.pq2, a.tensor)
n_in, n_out = int(dims[0]), int(dims[1])
T2, d2 = decode_pq2(raw, n_in * n_out)
T = torch.from_numpy(T2).to(dev).double().reshape(n_out, n_in)
dP = torch.from_numpy(d2).to(dev).double().reshape(n_out, n_in // G)
c = read_kv(a.pq2, "prism.hadamard.")
off = 0
for w in c["prism.hadamard.sign_widths"]:
    if w == n_in:
        s = torch.tensor(c["prism.hadamard.sign_values"][off:off + w], dtype=torch.float64, device=dev)
    off += w
with safe_open(a.base_st, "pt") as f:
    Wb = f.get_tensor(a.st_key).to(dev).double()
Wf = fwht_blocks(Wb * s, 1024)
X = fwht_blocks(torch.load(a.acts, map_location=dev).double() * s, 1024)
perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(0)).to(dev)
n_cal = int(0.75 * X.shape[0])
Xc, Xh = X[perm[:n_cal]], X[perm[n_cal:]]                 # calibration / HELD-OUT
Y0 = Xh @ Wf.T
H = Xc.T @ Xc / n_cal


def group_scale(W, k=86, mult=1.0):
    g = W.reshape(n_out, n_in // G, G).abs()
    top = g.topk(k, dim=2).values
    return mult * top.mean(2)                              # [out, groups]


def score(name, Q, R):
    Tq = torch.sign(Q)
    e = Xh @ Q.T - Y0
    r = {"out_rel_err_heldout": float(e.norm() / Y0.norm()),
         "zero_frac": float((Tq == 0).double().mean()),
         "groups_with_exactly_42_zeros": float(((Tq.reshape(-1, G) == 0).sum(1) == 42).double().mean()),
         "trit_agreement_with_prism": float((Tq == T).double().mean()),
         "weight_rel_dist_to_prism": float((Q - T * dP.repeat_interleave(G, 1)).norm() / (T * dP.repeat_interleave(G, 1)).norm())}
    R[name] = r
    print(f"{name:<52s} " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)


def gptq(W, d, damp=0.01, blocksize=128, topk=None):
    """Column-sequential GPTQ. d: [out, groups] scale. topk: if set, enforce exactly `topk`
    nonzeros per (row, group) by choosing, at the END of each group, the topk largest |target|."""
    W = W.clone()
    Q = torch.zeros_like(W)
    Hd = H + damp * H.diag().mean() * torch.eye(n_in, device=dev)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True)
    for i1 in range(0, n_in, blocksize):
        i2 = i1 + blocksize
        W1 = W[:, i1:i2].clone()
        Err = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        dg = d[:, i1 // G][:, None]
        if topk is not None:
            # budgeted: walk the group, but a weight may only be nonzero while budget remains and
            # must be nonzero once the remaining columns equal the remaining budget
            budget = torch.full((n_out,), topk, device=dev)
            # rank columns of this group by |w| to decide a provisional keep set, then compensate
            keep = torch.zeros_like(W1, dtype=torch.bool)
            keep.scatter_(1, W1.abs().topk(topk, dim=1).indices, True)
        for j in range(blocksize):
            w = W1[:, j]
            if topk is None:
                q = torch.round(w / dg[:, 0]).clamp(-1, 1) * dg[:, 0]
            else:
                left = blocksize - j
                must = budget >= left
                can = budget > 0
                want = (w.abs() > 0.5 * dg[:, 0])
                nzm = (want & can) | must
                q = torch.where(nzm, torch.sign(w) * dg[:, 0], torch.zeros_like(w))
                budget = budget - nzm.long()
            Q[:, i1 + j] = q
            err = (w - q) / Hi[j, j]
            W1[:, j:] -= err[:, None] * Hi[j, j:][None]
            Err[:, j] = err
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q


R = {}
dfull = dP.repeat_interleave(G, 1)
score("PRISM (shipped)", T * dfull, R)
d86 = group_scale(Wf)
g = Wf.reshape(n_out, -1, G)
m86 = torch.zeros_like(g, dtype=torch.bool).scatter_(2, g.abs().topk(86, dim=2).indices, True)
score("magnitude top-86, mean scale", (torch.sign(g) * m86 * d86[:, :, None]).reshape(n_out, n_in), R)
for mult in (1.0, 0.9, 0.8):
    for damp in (0.01, 0.1):
        score(f"GPTQ free zeros, d=top86*{mult}, damp={damp}", gptq(Wf, d86 * mult, damp), R)
for mult in (1.0, 0.9):
    score(f"GPTQ budget 86/group, d=top86*{mult}, damp=0.01", gptq(Wf, d86 * mult, 0.01, topk=86), R)
score("GPTQ free zeros with PRISM's own scales", gptq(Wf, dP, 0.01), R)
json.dump(R, open(a.out, "w"), indent=1)
print("GPTQ_DONE")
