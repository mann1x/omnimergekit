#!/usr/bin/env python3
"""bonsai_layer_lab.py — single-layer forensics + divergence bench for Bonsai 2 recovery.

One layer, three objects in ONE basis (PrismML's folded basis):
  Wf   the base model's layer, folded:  fwht(W * s)                      (full precision)
  P    PrismML's shipped layer, decoded: trits T in {-1,0,+1(,+2)} x per-128 scale d
  C_k  candidate ternarizations of Wf produced by OUR methods
and two questions per candidate:
  (a) DIVERGENCE  same input x, output of the ternary layer vs the original layer
  (b) IDENTITY    how close is the candidate to PrismML's actual trits and scales
Also decodes the same tensor from the PQ2_0 and the PTQ1_0 file and compares them trit-for-trit
(are the two packings the same values?).
"""
from __future__ import annotations

import argparse
import json
import os
import struct

import numpy as np
import torch
from safetensors import safe_open

from bonsai_hadamard_fold import fwht_blocks, read_kv

W = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
G = 128


def tensor_bytes(path: str, name: str):
    """(ggml_type, dims, raw bytes) of one tensor; manual walk, span from the offset table."""
    align = 32
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        _v, nt, nkv = struct.unpack("<IQQ", f.read(20))

        def rstr():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def skip(t):
            if t in W:
                return f.read(W[t])
            if t == 8:
                return rstr()
            (et,) = struct.unpack("<I", f.read(4))
            (cnt,) = struct.unpack("<Q", f.read(8))
            if et in W:
                f.seek(W[et] * cnt, 1)
            else:
                for _ in range(cnt):
                    skip(et)

        for _ in range(nkv):
            k = rstr()
            (t,) = struct.unpack("<I", f.read(4))
            v = skip(t)
            if k == "general.alignment":
                align = struct.unpack("<I", v)[0]
        ti = []
        for _ in range(nt):
            nm = rstr()
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            typ, off = struct.unpack("<IQ", f.read(12))
            ti.append((nm, typ, dims, off))
        base = (f.tell() + align - 1) // align * align
        offs = sorted({o for *_, o in ti}) + [os.path.getsize(path) - base]
        for nm, typ, dims, off in ti:
            if nm == name:
                n = int(np.prod(dims))
                size = {142: n // G * 34, 143: n // G * 28}.get(typ, offs[offs.index(off) + 1] - off)
                f.seek(base + off)
                return typ, dims, f.read(size)
    raise SystemExit(f"FATAL: {name} not in {path}")


def decode_pq2(raw: bytes, n: int):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n // G, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    qs = b[:, 2:]
    q = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(n // G, G).astype(np.int8) - 1
    return q, d


def decode_ptq1(raw: bytes, n: int):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n // G, 28)
    qs, qh = b[:, :24].astype(np.uint16), b[:, 24:26].astype(np.uint16)
    d = b[:, 26:28].copy().view(np.float16).astype(np.float32).reshape(-1)
    pow3 = [1, 3, 9, 27, 81]
    out, j = [], 0
    for c in (32, 16, 8):
        while j + c <= 24:
            for p in range(5):
                q = (qs[:, j:j + c] * pow3[p]) & 0xFF
                out.append(((q * 3) >> 8).astype(np.int8) - 1)
            j += c
    for p in range(4):
        q = (qh * pow3[p]) & 0xFF
        out.append(((q * 3) >> 8).astype(np.int8) - 1)
    return np.concatenate(out, axis=1), d


# ---------------- candidate ternarizers: (Wf groups [N,128]) -> (trits [N,128], scale [N]) ----------------
def t_codec_amax(g):
    """What the shipped quantize tool does: d = max|w|, t = round(w/d)."""
    d = g.abs().amax(1)
    return torch.round(g / d[:, None]).clamp(-1, 1), d


def t_twn(g, k=0.75):
    """Ternary Weight Networks: threshold k*mean|w|, scale = mean of the survivors."""
    a = g.abs()
    m = a > k * a.mean(1, keepdim=True)
    d = (a * m).sum(1) / m.sum(1).clamp(min=1)
    return torch.sign(g) * m, d


def t_opt(g):
    """Exact per-group L2 optimum: keep the top-k |w| with k maximizing (sum top-k)^2 / k."""
    a, _ = g.abs().sort(1, descending=True)
    cs = a.cumsum(1)
    ks = torch.arange(1, G + 1, device=g.device)
    k = (cs ** 2 / ks).argmax(1)
    thr = a.gather(1, k[:, None])
    m = g.abs() >= thr
    d = (g.abs() * m).sum(1) / m.sum(1).clamp(min=1)
    return torch.sign(g) * m, d


def t_fixed_nz(g, nz):
    """Keep exactly `nz` largest |w| per group (AbdelH2O: PrismML groups have a fixed zero count)."""
    a = g.abs()
    thr = a.kthvalue(G - nz + 1, dim=1, keepdim=True).values
    m = a >= thr
    d = (a * m).sum(1) / m.sum(1).clamp(min=1)
    return torch.sign(g) * m, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pq2", required=True)
    ap.add_argument("--ptq1", required=True)
    ap.add_argument("--base-st", required=True)
    ap.add_argument("--st-key", required=True)
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--acts", help=".pt of real layer inputs [N, in]; Gaussian is used if absent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    R = {"tensor": a.tensor}

    typ, dims, raw = tensor_bytes(a.pq2, a.tensor)
    assert typ == 142, typ
    n_in, n_out = int(dims[0]), int(dims[1])
    n = n_in * n_out
    T2, d2 = decode_pq2(raw, n)
    typ1, _, raw1 = tensor_bytes(a.ptq1, a.tensor)
    assert typ1 == 143, typ1
    T1, d1 = decode_ptq1(raw1, n)
    R["packings"] = {"trits_equal": bool((T1 == T2).all()), "trit_mismatches": int((T1 != T2).sum()),
                     "scales_equal": bool((d1 == d2).all()),
                     "bytes_pq2_0": len(raw), "bytes_ptq1_0": len(raw1),
                     "bpw_pq2_0": len(raw) * 8 / n, "bpw_ptq1_0": len(raw1) * 8 / n}
    print("[packings]", json.dumps(R["packings"]), flush=True)

    T = torch.from_numpy(T2).to(dev).float()            # [n/128, 128], row-major over [out, in]
    d = torch.from_numpy(d2).to(dev)
    P = (T * d[:, None]).reshape(n_out, n_in)

    c = read_kv(a.pq2, "prism.hadamard.")
    assert a.tensor in c["prism.hadamard.weight_names"]
    block, off, s = int(c["prism.hadamard.block_size"]), 0, None
    for w in c["prism.hadamard.sign_widths"]:
        if w == n_in:
            s = torch.tensor(c["prism.hadamard.sign_values"][off:off + w], dtype=torch.float32, device=dev)
        off += w
    with safe_open(a.base_st, "pt") as f:
        Wb = f.get_tensor(a.st_key).to(dev).float()
    assert tuple(Wb.shape) == (n_out, n_in), Wb.shape
    Wf = fwht_blocks(Wb * s, block)
    g = Wf.reshape(-1, G)

    # ---- forensics on PrismML's layer, in the folded basis ----
    nzc = (T != 0).sum(1)
    vals, cnts = torch.unique(T, return_counts=True)
    amax, amean = g.abs().amax(1), g.abs().mean(1)
    _, d_opt = t_opt(g)
    nz = T != 0
    sign_agree = float((torch.sign(g)[nz] == T[nz]).float().mean())
    # is the zero set the small-|w| set?  rank of |w| inside its group for zeros vs nonzeros
    rank = g.abs().argsort(1).argsort(1).float()
    zc = torch.unique(G - nzc, return_counts=True)
    top = sorted(zip(zc[1].tolist(), zc[0].tolist()), reverse=True)[:5]
    R["prism"] = {
        "shape_out_in": [n_out, n_in], "groups": int(g.shape[0]),
        "trit_hist": {str(int(v)): int(k) for v, k in zip(vals, cnts)},
        "zero_frac": float((T == 0).float().mean()),
        "zeros_per_group_top5 (count_of_groups, zeros)": top,
        "scale_mean": float(d.mean()), "scale_over_group_amax": float((d / amax).median()),
        "scale_over_group_mean_abs": float((d / amean).median()),
        "scale_over_L2opt_scale": float((d / d_opt).median()),
        "sign_agreement_on_nonzeros": sign_agree,
        "mean_rank_of_zeros(0..127)": float(rank[~nz].mean()), "mean_rank_of_nonzeros": float(rank[nz].mean()),
        "weight_rel_err ||P-Wf||/||Wf||": float((P - Wf).norm() / Wf.norm()),
        "cosine(P,Wf)": float((P * Wf).sum() / (P.norm() * Wf.norm())),
        "norm_ratio ||P||/||Wf||": float(P.norm() / Wf.norm()),
    }
    print("[prism]", json.dumps(R["prism"], indent=1), flush=True)

    # ---- divergence bench ----
    if a.acts:
        X = torch.load(a.acts, map_location=dev).float()
        R["inputs"] = f"real activations {tuple(X.shape)}"
    else:
        torch.manual_seed(0)
        X = torch.randn(4096, n_in, device=dev)
        R["inputs"] = "Gaussian N(0,1) [4096, in] — NOT the layer's real input distribution"
    Y0 = X @ Wb.T
    Xr = fwht_blocks(X * s, block)                       # what the runtime feeds a folded weight
    assert torch.allclose(Xr @ Wf.T, Y0, rtol=1e-3, atol=1e-3 * float(Y0.abs().mean())), "fold identity broken"

    def bench(name, Tc, dc):
        Q = (Tc * dc[:, None]).reshape(n_out, n_in)
        Y = Xr @ Q.T
        e = Y - Y0
        both = (Tc != 0) | (T != 0)
        r = {
            "out_rel_err": float(e.norm() / Y0.norm()),
            "out_cosine": float(torch.nn.functional.cosine_similarity(Y, Y0, dim=1).mean()),
            "weight_rel_err": float((Q - Wf).norm() / Wf.norm()),
            "zero_frac": float((Tc == 0).float().mean()),
            "trit_agreement_with_prism": float((Tc == T).float().mean()),
            "nonzero_jaccard_with_prism": float(((Tc != 0) & (T != 0)).sum() / both.sum()),
            "scale_ratio_to_prism_median": float((dc / d).median()),
            "weight_rel_dist_to_prism": float((Q - P).norm() / P.norm()),
        }
        R.setdefault("candidates", {})[name] = r
        print(f"[{name}]", json.dumps(r), flush=True)

    bench("PRISM (shipped layer)", T, d)
    bench("codec amax-RTN (the quantize tool)", *t_codec_amax(g))
    bench("TWN 0.75", *t_twn(g))
    bench("L2-optimal per group", *t_opt(g))
    med_nz = int(nzc.float().median())
    bench(f"fixed {med_nz} nonzeros/group (prism median)", *t_fixed_nz(g, med_nz))
    # PrismML's own mask and signs, only the scale refit to the base weights: isolates trit choice from scale choice
    dfit = (g * T).sum(1) / (T * T).sum(1).clamp(min=1)
    bench("prism trits + scale L2-refit to base", T, dfit)
    # ---- step 2: WHERE does PrismML deviate from plain magnitude top-K, and does the input explain it? ----
    Tk, _ = t_fixed_nz(g, med_nz)
    mp, mk = T != 0, Tk != 0
    kept_small, dropped_big = mp & ~mk, ~mp & mk          # prism kept a sub-threshold weight / dropped a supra one
    dist = rank - (G - med_nz)                             # >=0 : inside magnitude top-K ; <0 : outside
    col = torch.arange(n_in, device=dev).repeat(n_out).reshape(-1, G)   # input column of every weight
    e_raw = Xr.pow(2).mean(0)                              # per-column input energy, rotated basis
    energy = e_raw[col] / e_raw.mean()
    R["deviation_from_topK"] = {
        "K": med_nz,
        "disagreeing_trits_frac": float((T != Tk).float().mean()),
        "sign_flips_among_prism_nonzeros": int(((torch.sign(g) != T) & mp).sum()),
        "kept_small_count": int(kept_small.sum()), "dropped_big_count": int(dropped_big.sum()),
        "kept_small: median rank distance below the cut": float(dist[kept_small].median()),
        "dropped_big: median rank distance above the cut": float(dist[dropped_big].median()),
        "share of swaps within 8 ranks of the cut": float(((dist[kept_small | dropped_big]).abs() <= 8).float().mean()),
        "rotated input energy: max/mean over columns": float(e_raw.max() / e_raw.mean()),
        "rel. input energy of kept_small columns (1.0 = average)": float(energy[kept_small].mean()),
        "rel. input energy of dropped_big columns": float(energy[dropped_big].mean()),
        "rel. input energy of agreed-kept columns": float(energy[mp & mk].mean()),
    }
    print("[deviation]", json.dumps(R["deviation_from_topK"], indent=1), flush=True)

    # activation-aware candidates: saliency = |w| * sqrt(E[x^2]) (Wanda-style), same K
    sal = g.abs() * e_raw[col].sqrt()
    thr = sal.kthvalue(G - med_nz + 1, dim=1, keepdim=True).values
    ms = sal >= thr
    ds = (g.abs() * ms).sum(1) / ms.sum(1).clamp(min=1)
    bench(f"top-{med_nz} by |w|*sqrt(E[x^2]) (activation-aware mask)", torch.sign(g) * ms, ds)
    we = e_raw[col]
    dwe = (g.abs() * ms * we).sum(1) / (ms * we).sum(1).clamp(min=1e-20)
    bench("  same mask, energy-weighted scale", torch.sign(g) * ms, dwe)
    dpw = (g * T * we).sum(1) / (T * T * we).sum(1).clamp(min=1e-20)
    bench("prism trits + energy-weighted scale refit", T, dpw)
    json.dump(R, open(a.out, "w"), indent=1)
    print("LAB_DONE ->", a.out)


if __name__ == "__main__":
    main()
