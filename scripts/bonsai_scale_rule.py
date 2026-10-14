#!/usr/bin/env python
"""Which rule produced PrismML's per-group scale? One layer, rotated basis, base weights vs their trits.

For every (row, group of 128) compare their stored scale d with candidate rules computed from the BASE rotated
weights g and THEIR trit pattern T. A rule that is right gives ratio 1.00 with a narrow spread; a learned scale
gives a broad spread around whatever its mean is. Also tests whether the trits are a threshold function of d.
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
ap.add_argument("--acts")
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:1")
a = ap.parse_args()
dev = a.device

typ, dims, raw = tensor_bytes(a.pq2, a.tensor)
n_in, n_out = int(dims[0]), int(dims[1])
T2, d2 = decode_pq2(raw, n_in * n_out)
T = torch.from_numpy(T2).to(dev).double()
d = torch.from_numpy(d2).to(dev).double()
c = read_kv(a.pq2, "prism.hadamard.")
off, s = 0, None
for w in c["prism.hadamard.sign_widths"]:
    if w == n_in:
        s = torch.tensor(c["prism.hadamard.sign_values"][off:off + w], dtype=torch.float32, device=dev)
    off += w
with safe_open(a.base_st, "pt") as f:
    Wb = f.get_tensor(a.st_key).to(dev).float()
g = fwht_blocks(Wb * s, int(c["prism.hadamard.block_size"])).reshape(-1, G).double()
ag, nz = g.abs(), T != 0
nnz = nz.sum(1)
srt = ag.sort(1, descending=True).values


def stat(name, rule):
    r = (d / rule)
    r = r[torch.isfinite(r)]
    q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95], device=dev, dtype=r.dtype))
    out = dict(median=float(q[1]), p05=float(q[0]), p95=float(q[2]), rel_spread=float((q[2] - q[0]) / q[1]))
    print(f"  d / {name:<44} median {out['median']:.4f}   p05 {out['p05']:.4f}  p95 {out['p95']:.4f}   spread {out['rel_spread']:.3f}", flush=True)
    return out


R = {"tensor": a.tensor, "groups": int(g.shape[0])}
print(f"{a.tensor}: {g.shape[0]} groups", flush=True)
R["rules"] = {
    "L2opt_their_trits  sum(g*T)/nnz": stat("L2-opt for THEIR trits  sum(g*T)/nnz", (g * T).sum(1) / nnz),
    "mean|g| over their nonzeros": stat("mean|g| over THEIR nonzeros", (ag * nz).sum(1) / nnz),
    "mean|g| top-86 of base": stat("mean|g| over base top-86", srt[:, :86].mean(1)),
    "mean|g| all 128": stat("mean|g| over all 128", ag.mean(1)),
    "rms g all 128": stat("rms(g) over all 128", g.pow(2).mean(1).sqrt()),
    "rms top-86": stat("rms over base top-86", srt[:, :86].pow(2).mean(1).sqrt()),
    "median|g| top-86": stat("median|g| of base top-86", srt[:, :86].median(1).values),
    "2 * 86th largest |g|": stat("2 x (86th largest |g|)  [threshold = d/2]", 2 * srt[:, 85]),
    "2 * midpoint(86th,87th)": stat("2 x midpoint(86th, 87th largest)", srt[:, 85] + srt[:, 86]),
    "max|g|": stat("max|g|", srt[:, 0]),
}
# threshold structure: is T == round(g/d).clamp(-1,1) for base g?  and where does the cut sit in units of d
rt = torch.round(g / d[:, None]).clamp(-1, 1)
min_nz = torch.where(nz, ag, torch.full_like(ag, 1e9)).amin(1) / d
max_z = torch.where(~nz, ag, torch.zeros_like(ag)).amax(1) / d
R["threshold"] = {
    "agree with round(g_base/d)": float((rt == T).double().mean()),
    "median min|g| among nonzeros / d": float(min_nz.median()),
    "median max|g| among zeros / d": float(max_z.median()),
    "groups where zero-set is exactly the smallest-|g| set": float((max_z <= min_nz).double().mean()),
    "nnz histogram": {int(k): int(v) for k, v in zip(*torch.unique(nnz, return_counts=True))},
}
print(json.dumps(R["threshold"], indent=1), flush=True)

if a.acts:
    X = torch.load(a.acts, map_location=dev).double()
    Xf = fwht_blocks((X.float() * s), int(c["prism.hadamard.block_size"])).double()
    H = Xf.T @ Xf / Xf.shape[0]
    ng = n_in // G
    Tr, Wr = T.view(n_out, n_in), g.view(n_out, n_in)
    Tv = Tr.view(n_out, ng, G)
    A = torch.empty(n_out, ng, ng, device=dev, dtype=torch.float64)
    for k in range(ng):
        P = Tr[:, k * G:(k + 1) * G] @ H[k * G:(k + 1) * G, :]
        A[:, k, :] = (P.view(n_out, ng, G) * Tv).sum(-1)
    b = ((Wr @ H).view(n_out, ng, G) * Tv).sum(-1)
    eye = torch.eye(ng, device=dev, dtype=torch.float64)
    hopt = torch.linalg.solve(A + 1e-6 * A.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * eye, b).reshape(-1)
    # diagonal-only (per-column energy weighted) optimum
    e = H.diag().view(1, ng, G)
    dopt = ((Wr.view(n_out, ng, G) * Tv * e).sum(-1) / (Tv * Tv * e).sum(-1)).reshape(-1)
    R["rules"]["H-optimal joint (activation-aware)"] = stat("activation-optimal, joint over the row", hopt)
    R["rules"]["energy-weighted per group"] = stat("energy-weighted per-group optimum", dopt)

    def err(sc):
        Q = (Tv * sc.view(n_out, ng, 1)).view(n_out, n_in)
        D = Q - Wr
        return float(((D @ H) * D).sum() / ((Wr @ H) * Wr).sum())

    R["output_rel_err2"] = {"their scale": err(d), "L2-opt for their trits": err((g * T).sum(1) / nnz),
                            "activation-optimal": err(hopt), "their scale x1.05": err(d * 1.05), "their scale x0.95": err(d * 0.95)}
    print(json.dumps(R["output_rel_err2"], indent=1), flush=True)
json.dump(R, open(a.out, "w"), indent=1)
print("SCALE_RULE_DONE")
