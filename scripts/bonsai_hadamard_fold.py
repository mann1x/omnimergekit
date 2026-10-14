#!/usr/bin/env python3
"""bonsai_hadamard_fold.py — fold a qwen35 GGUF into the Bonsai 2 Hadamard basis.

Writes a full-precision GGUF that computes the SAME function as the source, but with every
`prism.hadamard.weight_names` tensor stored in the rotated basis and `token_embd` stored as
latent rows, plus the `prism.hadamard.*` metadata that makes the runtime undo it. A ternary
quantizer run on the OUTPUT then rounds in the basis PrismML's format assumes.

CONTRACT (read out of the runtime, upstream llama.cpp #29600 / opencoti patch 0588):
  folded matmul      y = W_s . H( s * perm(x) )        H = blockwise normalized Sylvester WHT
  latent lookup      h = s * H(z)
so, H being symmetric and orthonormal (H.H = I) and s*s = 1:
  W_s = H-along-input-axis( perm_cols(W) * s )        z = H( s * h )
i.e. ONE operation, `fwht(x * s)` over the last axis, for both kinds. `perm` is the GDN
tiled->grouped feature reorder and applies to `ssm_out` only (gdn_v_grouped).

The contract (names, block size, signs, widths) is COPIED from a reference Bonsai GGUF rather
than re-derived, so the result is structurally identical to PrismML's by construction. The
reference is read with a KV-only walker: gguf-py cannot open a PQ2_0/PTQ1_0 file.

The output is the lossless gate's subject: it must reproduce the source's perplexity on the
reference engine BEFORE anything is quantized from it.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time

import numpy as np
import torch

import gguf

W = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4),
     6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}


def read_kv(path: str, want_prefix: str) -> dict:
    """KV-only GGUF walk; never touches the tensor table."""
    out = {}
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        _ver, _nt, nkv = struct.unpack("<IQQ", f.read(20))

        def rstr():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            if t in W:
                fmt, n = W[t]
                return struct.unpack("<" + fmt, f.read(n))[0]
            if t == 8:
                return rstr()
            if t == 9:
                (et,) = struct.unpack("<I", f.read(4))
                (cnt,) = struct.unpack("<Q", f.read(8))
                if et in W:
                    fmt, n = W[et]
                    return list(struct.unpack("<%d%s" % (cnt, fmt), f.read(n * cnt)))
                return [rval(et) for _ in range(cnt)]
            raise ValueError(t)

        for _ in range(nkv):
            k = rstr()
            (t,) = struct.unpack("<I", f.read(4))
            v = rval(t)
            if k.startswith(want_prefix):
                out[k] = v
    return out


def fwht_blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    """Normalized Sylvester WHT over the last axis in independent blocks (natural order:
    H[r, c] = (-1)^popcount(r & c) / sqrt(block), which is what the runtime builds)."""
    shape = x.shape
    x = x.reshape(-1, shape[-1] // block, block)
    h = 1
    while h < block:
        x = x.reshape(x.shape[0], x.shape[1], block // (2 * h), 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2)
        h *= 2
    return (x.reshape(shape) / (block ** 0.5))


def ternarize(y: torch.Tensor, mode: str) -> torch.Tensor:
    """Return t*d per 128-weight group along the last axis, as float32 (zeros where t == 0).

    The values are emitted as exact +-d so that the PQ2_0/PTQ1_0 codec, whose own rule is
    `d = max|w|, t = round(w/d)`, PACKS them losslessly: max|w| is d and every w/d is -1, 0 or 1.
    That codec rule is a packer, not a ternarizer — applied to real-valued weights it zeroes ~83%
    of them (measured on Bonsai 2 blk.3.attn_k), which is what collapsed the first v6 files.
      top:K[:m]  keep the K largest |w| per group, d = m * mean|kept|   (PrismML ships K=86 in 97.6% of groups)
      l2opt      per-group exact L2 optimum (K chosen per group)
      twn:k      threshold k * mean|w|, d = mean|kept|
    """
    g = y.reshape(-1, 128)
    a = g.abs()
    kind, *arg = mode.split(":")
    mult = 1.0
    if kind == "top":
        k = int(arg[0])
        mult = float(arg[1]) if len(arg) > 1 else 1.0
        m = a >= a.kthvalue(128 - k + 1, dim=1, keepdim=True).values
    elif kind == "l2opt":
        srt, _ = a.sort(1, descending=True)
        k = (srt.cumsum(1) ** 2 / torch.arange(1, 129, device=y.device)).argmax(1)
        m = a >= srt.gather(1, k[:, None])
    elif kind == "twn":
        m = a > float(arg[0]) * a.mean(1, keepdim=True)
    else:
        raise SystemExit(f"FATAL: unknown --ternarize mode {mode}")
    d = mult * (a * m).sum(1) / m.sum(1).clamp(min=1)
    # snap d to bf16 BEFORE use so the written +-d and the codec's fp16 scale are the same number
    bits = d.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    d = bits.to(torch.int32).view(torch.float32)
    return (torch.sign(g) * m * d[:, None]).reshape(y.shape)


def f32_to_bf16(x: torch.Tensor) -> np.ndarray:
    bits = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return bits.to(torch.int32).cpu().numpy().astype(np.uint16)


def load_f32(t, dev) -> torch.Tensor:
    qt = t.tensor_type
    a = np.asarray(t.data)
    if qt == gguf.GGMLQuantizationType.BF16:
        # the caller reshapes; go through raw bytes so the reader's dtype/shape choice is irrelevant
        u = torch.from_numpy(a.reshape(-1).view(np.uint16).astype(np.int32)).to(dev)
        return (u << 16).view(torch.float32)
    if qt == gguf.GGMLQuantizationType.F16:
        return torch.from_numpy(a.view(np.float16).copy()).to(dev).float()
    if qt == gguf.GGMLQuantizationType.F32:
        return torch.from_numpy(a.view(np.float32).copy()).to(dev)
    raise SystemExit(f"FATAL: {t.name} is {qt.name}; the source must be full precision")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="full-precision source GGUF (BF16/F16/F32)")
    ap.add_argument("--ref", required=True, help="reference Bonsai GGUF to copy the contract from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ternarize", help="top:K[:mult] | l2opt | twn:k — ternarize the contract tensors "
                    "after the fold; the result is still a bf16 GGUF, exactly packable by the codec")
    ap.add_argument("--override-dir", help="dir of <tensor name>.pt files holding ALREADY-FOLDED bf16 values in "
                    "GGUF layout (from bonsai_pt2_full.py); every contract tensor must be present")
    ap.add_argument("--mtp-head", help="GGUF from `convert_hf_to_gguf --mtp`; its blk.N.* tensors are "
                    "appended UNFOLDED (the head is not in the contract) and block_count is bumped")
    a = ap.parse_args()

    c = read_kv(a.ref, "prism.hadamard.")
    need = ["version", "block_size", "transform", "axis", "sign_mode", "sign_widths",
            "sign_values", "weight_names", "inverse_weight_names"]
    for k in need:
        if "prism.hadamard." + k not in c:
            raise SystemExit(f"FATAL: reference lacks prism.hadamard.{k}")
    P = lambda k: c["prism.hadamard." + k]  # noqa: E731
    assert P("transform") == "normalized-sylvester-walsh-hadamard" and P("axis") == "input-last-dimension"
    assert P("sign_mode") == "explicit" and P("version") == 1, "only the v1/explicit contract is implemented"
    block = int(P("block_size"))
    signs, off = {}, 0
    for w in P("sign_widths"):
        signs[w] = torch.tensor(P("sign_values")[off:off + w], dtype=torch.float32, device=a.device)
        off += w
    assert off == len(P("sign_values"))
    fold = set(P("weight_names")) | set(P("inverse_weight_names"))
    grouped = bool(c.get("prism.hadamard.gdn_v_grouped", False))
    print(f"[contract] block={block} widths={list(signs)} fold={len(fold)} gdn_v_grouped={grouped}", flush=True)

    rd = gguf.GGUFReader(a.src)
    kv = {f.name: f for f in rd.fields.values()}
    prefolded = "prism.hadamard.version" in kv
    if prefolded:
        mine = read_kv(a.src, "prism.hadamard.")
        if mine != c:
            raise SystemExit("FATAL: the source is folded under a DIFFERENT contract than --ref")
        if not a.ternarize:
            raise SystemExit("FATAL: the source is already folded and no --ternarize was asked: nothing to do")
        print("[contract] source is already folded under this contract: fold step skipped", flush=True)
    names = {t.name for t in rd.tensors}
    missing = sorted(fold - names)
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} contract tensors absent from the source, e.g. {missing[:3]}")
    arch = kv["general.architecture"].contents()

    perm = None
    if grouped:
        n_v = int(kv[f"{arch}.ssm.time_step_rank"].contents())
        n_k = int(kv[f"{arch}.ssm.group_count"].contents())
        width = int(kv[f"{arch}.ssm.inner_size"].contents())
        hd, rep = width // n_v, n_v // n_k
        # runtime: x_grouped[i + hd*(r + rep*k)] = x[i + hd*(k + n_k*r)]
        g = torch.arange(width, device=a.device)
        i, rk = g % hd, g // hd
        r, k = rk % rep, rk // rep
        perm = i + hd * (k + n_k * r)
        assert torch.equal(torch.sort(perm).values, g), "GDN permutation is not a bijection"
        print(f"[contract] GDN perm: width={width} hd={hd} n_k={n_k} rep={rep}", flush=True)

    head, head_kv = [], {}
    if a.mtp_head:
        hr = gguf.GGUFReader(a.mtp_head)
        n_blk = int(kv[f"{arch}.block_count"].contents())
        head = [t for t in hr.tensors if t.name.startswith(f"blk.{n_blk}.")]
        stray = [t.name for t in hr.tensors if t.name.startswith("blk.") and t not in head]
        if not head or stray or any(t.name in names for t in head):
            raise SystemExit(f"FATAL: MTP head does not sit at blk.{n_blk} (head={len(head)} stray={stray[:3]})")
        hk = {f.name: f for f in hr.fields.values()}
        head_kv = {f"{arch}.block_count": int(hk[f"{arch}.block_count"].contents()),
                   f"{arch}.nextn_predict_layers": int(hk[f"{arch}.nextn_predict_layers"].contents())}
        assert head_kv[f"{arch}.block_count"] == n_blk + head_kv[f"{arch}.nextn_predict_layers"]
        print(f"[mtp] grafting {len(head)} head tensors at blk.{n_blk}; {head_kv}", flush=True)

    wr = gguf.GGUFWriter(a.out, arch)
    for f in rd.fields.values():
        if f.name.startswith("GGUF.") or f.name == "general.architecture":
            continue
        if f.name.startswith("prism.hadamard."):
            continue  # an already-folded source: its contract is re-emitted below, verbatim
        if f.name in head_kv:
            continue
        vt = f.types[0]
        sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        wr.add_key_value(f.name, f.contents(), vt, sub_type=sub)
    wr.add_uint32("prism.hadamard.version", 1)
    wr.add_string("prism.hadamard.transform", P("transform"))
    wr.add_uint32("prism.hadamard.block_size", block)
    wr.add_string("prism.hadamard.axis", P("axis"))
    wr.add_string("prism.hadamard.sign_mode", "explicit")
    wr.add_key_value("prism.hadamard.sign_widths", [int(x) for x in P("sign_widths")],
                     gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT32)
    wr.add_key_value("prism.hadamard.sign_values", [int(x) for x in P("sign_values")],
                     gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT32)
    wr.add_array("prism.hadamard.weight_names", list(P("weight_names")))
    wr.add_array("prism.hadamard.inverse_weight_names", list(P("inverse_weight_names")))
    wr.add_bool("prism.hadamard.gdn_v_grouped", grouped)
    for k, v in head_kv.items():
        wr.add_uint32(k, v)

    for t in rd.tensors:
        shp = tuple(reversed([int(d) for d in t.shape]))
        if t.name in fold:
            wr.add_tensor_info(t.name, shp, np.dtype(np.uint16), int(np.prod(shp)) * 2,
                               raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            # pass the reader's OWN shape+dtype: for BF16 that is a uint8 byte shape, which the writer
            # converts itself. Handing it a logical shape with a uint8 dtype halves the dims (bug-756).
            d = np.asarray(t.data)
            wr.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, raw_dtype=t.tensor_type)
    for t in head:
        d = np.asarray(t.data)
        wr.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, raw_dtype=t.tensor_type)
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_ti_data_to_file()

    t0, n_fold, n_perm, worst = time.time(), 0, 0, 0.0
    for idx, t in enumerate(rd.tensors):
        if t.name not in fold:
            wr.write_tensor_data(np.asarray(t.data))
            continue
        shp = tuple(reversed([int(d) for d in t.shape]))
        if a.override_dir:
            q = torch.load(f"{a.override_dir}/{t.name}.pt")
            if tuple(q.shape) != shp or q.dtype != torch.bfloat16:
                raise SystemExit(f"FATAL: override {t.name} is {tuple(q.shape)} {q.dtype}, expected {shp} bfloat16")
            wr.write_tensor_data(q.view(torch.int16).numpy().view(np.uint16))
            n_fold += 1
            continue
        x = load_f32(t, a.device).reshape(shp)
        width = shp[-1]
        if width not in signs:
            raise SystemExit(f"FATAL: no sign vector for width {width} ({t.name})")
        if prefolded:
            y = x
        else:
            if perm is not None and ".ssm_out." in t.name:
                x = x[..., perm]
                n_perm += 1
            y = fwht_blocks(x * signs[width], block)
            # orthonormality self-check: the fold must preserve every row's norm
            rel = float(((y.norm(dim=-1) - x.norm(dim=-1)).abs().max() / x.norm(dim=-1).max()))
            worst = max(worst, rel)
        if a.ternarize:
            y = ternarize(y, a.ternarize)
        wr.write_tensor_data(f32_to_bf16(y).reshape(shp))
        n_fold += 1
        del x, y
        if n_fold % 40 == 0:
            print(f"  [{idx + 1}/{len(rd.tensors)}] folded {n_fold}  {time.time() - t0:.0f}s", flush=True)
    for t in head:
        wr.write_tensor_data(np.asarray(t.data))
    wr.close()
    print(f"[done] folded={n_fold} (expected {len(fold)}) gdn-permuted={n_perm} "
          f"worst row-norm drift={worst:.2e}  {time.time() - t0:.0f}s", flush=True)
    if n_fold != len(fold) or worst > 1e-3:
        sys.exit("FATAL: fold count or orthonormality check failed")
    print("FOLD_OK")


if __name__ == "__main__":
    main()
