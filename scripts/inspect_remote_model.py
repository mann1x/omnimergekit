#!/usr/bin/env python3
"""Inspect a published model's ACTUAL WEIGHTS without downloading it.

Handles GGUF, safetensors, and LoRA adapters.

A GGUF header carries every tensor's name, shape, type and byte OFFSET. That
means you can parse the header over HTTP Range, then range-fetch individual
tensors and compare them BIT-EXACTLY against a reference HF model. Answering
"is this published model really a fine-tune, or is it the base re-uploaded?"
cost 14.3 MB against a 17.92 GB file the first time this was used.

    # what's in it
    python scripts/inspect_remote_gguf.py prithivMLmods/OxCoder-9B-GGUF:OxCoder-9B.BF16.gguf

    # is it really different from the base it claims to fine-tune?
    python scripts/inspect_remote_gguf.py prithivMLmods/OxCoder-9B-GGUF:OxCoder-9B.BF16.gguf \
        --ref /models/Qwen3.5-9B

    # the STRONG form: two references disambiguate converter from content
    python scripts/inspect_remote_gguf.py <gguf> --ref /models/Qwen3.5-9B --ref /models/OxCoder-9B

WHY THE TWO-REFERENCE FORM MATTERS
----------------------------------
A naive GGUF-vs-HF diff LIES. On the run that motivated this tool a flat
comparison reported 201 of 331 tensors "differing", of which exactly 24 were
real. The rest were converter conventions:

  * every `*_norm.weight`         GGUF = HF + 1     (RMSNorm offset convention)
  * `ssm_a`                       GGUF = -exp(A_log)
  * `ssm_conv1d/dt/out/alpha/...` layout permutation of the same values

With TWO references A and B that are bit-identical to each other on a tensor,
a GGUF differing from BOTH can only be the converter -- content cannot make a
file differ from two things that are equal. That test is cheap (it works on a
byte PREFIX) and needs no per-architecture transform table, so it stays correct
on architectures this file has never seen.

With ONE reference the known transforms below are applied first, and anything
whose family is layout-permuted is reported as UNVERIFIED rather than silently
counted as a difference -- pass `--full` (or a second `--ref`) to settle those.

The verdict is weighed by PARAMETERS, never by tensor count. On the motivating
run the 24 content-differing tensors were 5.62% of tensors but 0.000034% of
parameters (3,072 of 8.95 B) -- a tensor-count gate would have waved through a
model that is its own base with 24 norm vectors nudged by under one bf16 ULP.

Exit codes: 0 ok; 2 the differing PARAMETER share was below --min-delta-pct
(i.e. the GGUF is effectively the reference); 1 on error.
"""
from __future__ import annotations

import argparse
import json
import http.client
import os
import re
import struct
import urllib.parse
import urllib.request
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

# --------------------------------------------------------------------------
# GGUF primitives
# --------------------------------------------------------------------------
GGUF_MAGIC = b"GGUF"
_FMT = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
        5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
        12: ("<d", 8)}
_TNAME = {0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16", 4: "UINT32",
          5: "INT32", 6: "FLOAT32", 7: "BOOL", 8: "STRING", 9: "ARRAY",
          10: "UINT64", 11: "INT64", 12: "FLOAT64"}
# Only the un-quantised types can be compared elementwise against HF weights.
_GGML = {0: ("F32", np.dtype("<f4"), 4), 1: ("F16", np.dtype("<f2"), 2),
         30: ("BF16", None, 2)}


class Source:
    """Byte source for a GGUF: an HTTP URL (Range requests) or a local file."""

    def __init__(self, ref: str, token: str | None = None):
        self.url: str | None = None
        self.path: str | None = None
        self.token = token
        self.fetched = 0
        self.requests = 0
        self._conn: Any = None
        self._host = self._sel = self._scheme = ""
        self._hdrs: dict[str, str] = {}
        if os.path.exists(ref):
            self.path = ref
        elif ref.startswith("http://") or ref.startswith("https://"):
            self.url = ref
        else:
            # "owner/repo:file"  or  "owner/repo/file"
            if ":" in ref:
                repo, _, fn = ref.partition(":")
            else:
                parts = ref.split("/")
                if len(parts) < 3:
                    raise SystemExit(
                        f"cannot parse GGUF ref {ref!r}: use a path, a URL, or "
                        f"owner/repo:file.gguf")
                repo, fn = "/".join(parts[:2]), "/".join(parts[2:])
            self.url = (f"https://huggingface.co/{repo}/resolve/main/{fn}")

    def _connect(self) -> None:
        """Resolve the CDN redirect ONCE and hold a keep-alive connection.

        Re-resolving per request is what made the first version unusable: 760
        tensors meant 760 redirect chains and 760 TLS handshakes, so a 48 MB
        read took longer than downloading the whole 18 GB file. The bytes were
        never the bottleneck -- the round trips were.
        """
        hdrs = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        req = urllib.request.Request(self.url, headers={**hdrs,
                                                        "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            final = r.geturl()
        u = urllib.parse.urlsplit(final)
        self._host, self._scheme = u.netloc, u.scheme
        self._sel = u.path + ("?" + u.query if u.query else "")
        # A signed CDN URL carries its own auth; re-sending ours can 400.
        self._hdrs = {} if u.netloc != "huggingface.co" else hdrs
        self._conn = (http.client.HTTPSConnection(self._host, timeout=180)
                      if self._scheme == "https"
                      else http.client.HTTPConnection(self._host, timeout=180))

    def read(self, start: int, n: int) -> bytes:
        if n <= 0:
            return b""
        if self.path:
            with open(self.path, "rb") as h:
                h.seek(start)
                b = h.read(n)
            self.fetched += len(b)
            return b
        last: Exception | None = None
        for attempt in range(4):
            try:
                if self._conn is None:
                    self._connect()
                self._conn.request("GET", self._sel, headers={
                    **self._hdrs, "Range": f"bytes={start}-{start + n - 1}"})
                r = self._conn.getresponse()
                b = r.read()
                if r.status not in (200, 206):
                    raise OSError(f"HTTP {r.status}")
                self.fetched += len(b)
                self.requests += 1
                return b
            except Exception as e:  # noqa: BLE001 - reconnect and retry
                last = e
                try:
                    if self._conn:
                        self._conn.close()
                except Exception:  # noqa: BLE001
                    pass
                self._conn = None
                if attempt == 2:
                    self.url = self.url  # force a fresh redirect resolve
        raise SystemExit(f"range fetch failed at {start}+{n}: {last}")


class _Header:
    """Incrementally-grown buffer over the front of the file."""

    def __init__(self, src: Source):
        self.src, self.buf = src, b""

    def at(self, off: int, n: int) -> bytes:
        while len(self.buf) < off + n:
            want = max(off + n, len(self.buf) + (1 << 20))
            self.buf += self.src.read(len(self.buf), want - len(self.buf))
        return self.buf[off:off + n]


def _rstr(h: _Header, o: int) -> tuple[str, int]:
    n = struct.unpack("<Q", h.at(o, 8))[0]
    return h.at(o + 8, n).decode("utf-8", "replace"), o + 8 + n


def _rval(h: _Header, o: int, t: int) -> tuple[Any, int]:
    if t == 8:
        return _rstr(h, o)
    if t == 9:
        et = struct.unpack("<I", h.at(o, 4))[0]
        n = struct.unpack("<Q", h.at(o + 4, 8))[0]
        o += 12
        head = []
        for i in range(n):
            v, o = _rval(h, o, et)
            if i < 8:
                head.append(v)
        return {"array": _TNAME.get(et, et), "len": n, "head": head}, o
    fmt, sz = _FMT[t]
    return struct.unpack(fmt, h.at(o, sz))[0], o + sz


def read_header(src: Source) -> dict:
    h = _Header(src)
    if h.at(0, 4) != GGUF_MAGIC:
        raise SystemExit("not a GGUF file (bad magic)")
    ver, = struct.unpack("<I", h.at(4, 4))
    ntensor, = struct.unpack("<Q", h.at(8, 8))
    nkv, = struct.unpack("<Q", h.at(16, 8))
    o, kv = 24, {}
    for _ in range(nkv):
        k, o = _rstr(h, o)
        t, = struct.unpack("<I", h.at(o, 4))
        o += 4
        kv[k], o = _rval(h, o, t)
    infos = []
    for _ in range(ntensor):
        name, o = _rstr(h, o)
        nd, = struct.unpack("<I", h.at(o, 4))
        o += 4
        dims = list(struct.unpack("<" + "Q" * nd, h.at(o, 8 * nd)))
        o += 8 * nd
        tt, = struct.unpack("<I", h.at(o, 4))
        o += 4
        off, = struct.unpack("<Q", h.at(o, 8))
        o += 8
        infos.append({"name": name, "dims": dims, "type": tt, "offset": off})
    align = int(kv.get("general.alignment", 32) or 32)
    return {"version": ver, "kv": kv, "infos": infos,
            "data_start": (o + align - 1) // align * align,
            "header_bytes": o}


_ST_DTYPE = {"F32": 0, "F16": 1, "BF16": 30}


def read_st_header(src: Source) -> dict | None:
    """Parse a safetensors header: u64 length, then JSON {name:{dtype,shape,
    data_offsets}}. Returns the same shape of dict as read_header(), so the
    comparison path below is format-agnostic. None if this is not safetensors."""
    n = struct.unpack("<Q", src.read(0, 8))[0]
    if not (0 < n < (1 << 30)):
        return None
    try:
        js = json.loads(src.read(8, n))
    except Exception:  # noqa: BLE001 - not safetensors
        return None
    js.pop("__metadata__", None)
    infos = []
    for name, e in js.items():
        if "data_offsets" not in e or e.get("dtype") not in _ST_DTYPE:
            continue
        infos.append({"name": name, "dims": list(e["shape"]),
                      "type": _ST_DTYPE[e["dtype"]],
                      "offset": e["data_offsets"][0],
                      "nbytes": e["data_offsets"][1] - e["data_offsets"][0]})
    return {"version": 0, "kv": {"general.architecture": "safetensors"},
            "infos": infos, "data_start": 8 + n, "header_bytes": 8 + n,
            "format": "safetensors"}


def read_tensor(src: Source, hdr: dict, info: dict,
                max_bytes: int | None) -> torch.Tensor | None:
    """Fetch a tensor (or its byte prefix). None if the type is quantised."""
    spec = _GGML.get(info["type"])
    if spec is None:
        return None
    _, npdt, esz = spec
    total = int(info.get("nbytes") or (int(np.prod(info["dims"])) * esz))
    n = total if max_bytes in (None, 0) else min(max_bytes, total)
    n -= n % esz
    raw = src.read(hdr["data_start"] + info["offset"], n)
    if npdt is None:  # BF16: reinterpret via uint8
        return torch.from_numpy(
            np.frombuffer(raw, dtype=np.uint8).copy()).view(torch.bfloat16)
    return torch.from_numpy(np.frombuffer(raw, dtype=npdt).copy())


# --------------------------------------------------------------------------
# GGUF <-> HF name mapping and known converter transforms
# --------------------------------------------------------------------------
# transform: "identity" | "plus1" | "negexp" | "layout"
#   plus1  -- GGUF stores (HF + 1); the RMSNorm offset convention.
#   negexp -- GGUF stores -exp(HF); Mamba/SSM A_log.
#   layout -- same VALUES, permuted/reshaped; only a multiset test is valid,
#             and that needs the whole tensor (see --full).
_QWEN35 = {
    "attn_norm.weight":           ("input_layernorm.weight", "plus1"),
    "post_attention_norm.weight": ("post_attention_layernorm.weight", "plus1"),
    "attn_q_norm.weight":         ("self_attn.q_norm.weight", "plus1"),
    "attn_k_norm.weight":         ("self_attn.k_norm.weight", "plus1"),
    "attn_q.weight":              ("self_attn.q_proj.weight", "identity"),
    "attn_k.weight":              ("self_attn.k_proj.weight", "identity"),
    "attn_v.weight":              ("self_attn.v_proj.weight", "identity"),
    "attn_output.weight":         ("self_attn.o_proj.weight", "identity"),
    "ffn_up.weight":              ("mlp.up_proj.weight", "identity"),
    "ffn_down.weight":            ("mlp.down_proj.weight", "identity"),
    "ffn_gate.weight":            ("mlp.gate_proj.weight", "identity"),
    "ssm_norm.weight":            ("linear_attn.norm.weight", "identity"),
    "ssm_a":                      ("linear_attn.A_log", "negexp"),
    "ssm_dt.bias":                ("linear_attn.dt_bias", "layout"),
    "ssm_conv1d.weight":          ("linear_attn.conv1d.weight", "layout"),
    "ssm_out.weight":             ("linear_attn.out_proj.weight", "layout"),
    "ssm_alpha.weight":           ("linear_attn.in_proj_a.weight", "layout"),
    "ssm_beta.weight":            ("linear_attn.in_proj_b.weight", "layout"),
    "attn_qkv.weight":            ("linear_attn.in_proj_qkv.weight", "layout"),
    "attn_gate.weight":           ("linear_attn.in_proj_z.weight", "layout"),
}
_GLOBAL = {
    "token_embd.weight": ("embed_tokens.weight", "identity"),
    "output.weight":     ("lm_head.weight", "identity"),
    "output_norm.weight": ("norm.weight", "plus1"),
}
ARCH_MAPS = {"qwen35": _QWEN35, "qwen3": _QWEN35, "qwen2": _QWEN35,
             "llama": _QWEN35}


def hf_candidates(gname: str, arch: str) -> tuple[list[str], str]:
    """GGUF tensor name -> candidate HF names (prefix variants), + transform."""
    smap = ARCH_MAPS.get(arch, _QWEN35)
    m = re.match(r"blk\.(\d+)\.(.+)$", gname)
    if m:
        idx, suf = m.group(1), m.group(2)
        if suf not in smap:
            return [], "unknown"
        hf, tf = smap[suf]
        return ([f"model.language_model.layers.{idx}.{hf}",
                 f"model.layers.{idx}.{hf}"], tf)
    if gname in _GLOBAL:
        hf, tf = _GLOBAL[gname]
        base = hf.rsplit(".", 1)[0] if hf.startswith("lm_head") else hf
        cands = [f"model.language_model.{hf}", f"model.{hf}", hf]
        if hf.startswith("lm_head"):
            cands = ["lm_head.weight", "model.lm_head.weight"]
        del base
        return cands, tf
    return [], "unknown"


class Ref:
    """A local HF safetensors model directory."""

    def __init__(self, path: str):
        import glob
        self.path, self.index, self._h = path, {}, {}
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not files:
            raise SystemExit(f"no .safetensors in {path}")
        for f in files:
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    self.index[k] = f

    def get(self, key: str) -> torch.Tensor:
        f = self.index[key]
        if f not in self._h:
            self._h[f] = safe_open(f, "pt")
        return self._h[f].get_tensor(key)

    def resolve(self, cands: list[str]) -> str | None:
        for c in cands:
            if c in self.index:
                return c
        return None


def _apply(t: torch.Tensor, tf: str) -> torch.Tensor:
    if tf == "plus1":
        return t.float() + 1.0
    if tf == "negexp":
        return -torch.exp(t.float())
    return t


def _match(g: torch.Tensor, r: torch.Tensor, tf: str,
           full: bool) -> tuple[bool, str, float]:
    """Compare a GGUF tensor against a (transformed) reference tensor."""
    r = _apply(r, tf)
    r = r.float() if g.dtype == torch.float32 else r
    flat = r.flatten()
    if tf == "layout":
        if not full:
            return False, "unverified", float("nan")
        a = torch.sort(g.flatten().float())[0]
        b = torch.sort(flat.float())[0]
        if a.numel() == b.numel() and torch.equal(a, b):
            return True, "layout", 0.0
        return False, "differs", float("nan")
    sub = flat[:g.numel()]
    if g.numel() != sub.numel():
        return False, "shape", float("nan")
    # NEVER downcast the reference to the file's dtype. Doing so rounds the
    # reference into the file's grid and ERASES exactly the sub-ULP differences
    # this tool exists to find: a model re-saved from F32 to BF16 then reports
    # as bit-identical to its own base. Compare in the HIGHER precision, and
    # report a precision-only match as its own verdict.
    if g.dtype == sub.dtype and torch.equal(g, sub):
        return True, "identity" if tf == "identity" else tf, 0.0
    gf, rf = g.float(), sub.float()
    if torch.equal(gf, rf):
        return True, "identity" if tf == "identity" else tf, 0.0
    if g.dtype != sub.dtype and torch.equal(g, sub.to(g.dtype)):
        return True, "downcast", float((gf - rf).abs().max())
    return False, "differs", float((gf - rf).abs().max())



# --------------------------------------------------------------------------
# LoRA adapters: a different failure mode needs a different gate
# --------------------------------------------------------------------------
def analyse_adapter(src: Source, hdr: dict, thresh: float) -> int:
    """A LoRA adapter cannot be compared to a base -- it IS the delta.

    Its characteristic failure is `lora_B` still full of zeros: B initialises to
    zero so that BA = 0 at step 0, which means an adapter saved before training
    (or saved from the wrong object) applies EXACTLY NOTHING while looking
    perfectly well-formed -- right rank, right targets, right file size.
    So the gate is: how many B matrices are non-trivially non-zero.
    """
    infos = {i["name"]: i for i in hdr["infos"]}
    pairs: dict[str, dict] = {}
    for n in infos:
        if ".lora_A" in n or ".lora_B" in n:
            key = n.split(".lora_")[0]
            pairs.setdefault(key, {})["A" if ".lora_A" in n else "B"] = n
    other = [n for n in infos if ".lora_" not in n]
    print(f"\nLoRA adapter: {len(pairs)} A/B pairs"
          + (f", {len(other)} non-LoRA tensors" if other else ""))
    if not pairs:
        print("  no lora_A/lora_B tensors found -- not a LoRA adapter?")
        return 1
    zero_b = 0
    rows = []
    for key in sorted(pairs):
        p = pairs[key]
        if "A" not in p or "B" not in p:
            continue
        A = read_tensor(src, hdr, infos[p["A"]], None)
        B = read_tensor(src, hdr, infos[p["B"]], None)
        if A is None or B is None:
            continue
        Af, Bf = A.float(), B.float()
        bn, an = float(Bf.norm()), float(Af.norm())
        if bn == 0.0:
            zero_b += 1
        rows.append((key, an, bn, float(Bf.abs().max())))
    n = len(rows)
    live = n - zero_b
    print(f"  rank/shape sample : A{list(read_tensor(src, hdr, infos[pairs[sorted(pairs)[0]]['A']], None).shape)}")
    print(f"  pairs measured    : {n}")
    print(f"  lora_B ALL-ZERO   : {zero_b}   <- these apply nothing")
    print(f"  lora_B non-zero   : {live}")
    if rows:
        bns = sorted(r[2] for r in rows)
        print(f"  ||B||_F  min/med/max : {bns[0]:.4g} / {bns[len(bns)//2]:.4g} / {bns[-1]:.4g}")
        print("\n  sample targets:")
        for k, an, bn, bmax in rows[:8]:
            print("    %-56s ||A||=%-10.4g ||B||=%-10.4g max|B|=%.4g"
                  % (k.replace("base_model.model.", "")[:56], an, bn, bmax))
    frac = 100.0 * live / n if n else 0.0
    print(f"\nlive adapter fraction: {live}/{n} ({frac:.2f}%)")
    if frac < thresh:
        print("REFUSE: lora_B is zero -- this adapter applies NOTHING to the base.")
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Inspect / verify a GGUF's weights without downloading it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit 2 = real differences fell below --min-delta-pct")
    ap.add_argument("gguf", help="GGUF or safetensors: local path, URL, or owner/repo:file")
    ap.add_argument("--ref", action="append", default=[], metavar="DIR",
                    help="local HF model dir to compare against (repeatable; "
                         "TWO refs enable the converter-vs-content test)")
    ap.add_argument("--sample-bytes", type=int, default=64 * 1024,
                    help="bytes per large tensor (default 65536)")
    ap.add_argument("--full", action="store_true",
                    help="fetch whole tensors: exact for layout families, but "
                         "downloads most of the file")
    ap.add_argument("--tensors", metavar="REGEX",
                    help="only tensors whose name matches")
    ap.add_argument("--max-tensors", type=int, default=48, metavar="N",
                    help="sample at most N tensors, STRATIFIED across families "
                         "(default 48; 0 = every tensor). A real fine-tune moves "
                         "essentially every weight, so a stratified sample "
                         "answers 'is this the base?' in seconds")
    ap.add_argument("--min-delta-pct", type=float, default=None,
                    help="exit 2 if real-differing tensors are under this %% "
                         "of compared tensors (a fine-tune gate)")
    ap.add_argument("--json", metavar="OUT", help="write a JSON report")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    a = ap.parse_args()

    src = Source(a.gguf, token=os.environ.get("HF_TOKEN"))
    if src.read(0, 4) == GGUF_MAGIC:
        hdr = read_header(src)
    else:
        hdr = read_st_header(src)
        if hdr is None:
            raise SystemExit("not a GGUF and not a safetensors file")
    kv = hdr["kv"]
    arch = str(kv.get("general.architecture", "?"))
    infos = hdr["infos"]
    if a.tensors:
        rx = re.compile(a.tensors)
        infos = [i for i in infos if rx.search(i["name"])]
    n_all = len(infos)
    if a.max_tensors and n_all > a.max_tensors:
        # Stratify by tensor FAMILY (the suffix after blk.N.), then round-robin.
        # A flat head/prefix slice would sample one layer deeply and miss a
        # fine-tune that only moved, say, the MLPs -- the family is the axis
        # along which training is uneven, so it is the axis to balance on.
        fams: dict[str, list] = {}
        for i in infos:
            m = re.match(r"blk\.\d+\.(.+)$|^(.*)$", i["name"])
            fams.setdefault(m.group(1) or m.group(2), []).append(i)
        order, keys = [], sorted(fams)
        for j in range(max(len(v) for v in fams.values())):
            for k in keys:
                if j < len(fams[k]):
                    order.append(fams[k][j])
        infos = order[:a.max_tensors]
        print(f"  SAMPLED {len(infos)} of {n_all} tensors, stratified across "
              f"{len(keys)} families (--max-tensors 0 for all)")

    print(f"GGUF   : {src.path or src.url}")
    print(f"  arch={arch}  version={hdr['version']}  tensors={len(hdr['infos'])}"
          f"  header={hdr['header_bytes'] / 1e6:.1f} MB")
    for k in ("general.name", "general.basename", "general.size_label",
              "general.base_model.0.name", "general.file_type"):
        if k in kv:
            print(f"  {k:28s} {str(kv[k])[:64]}")
    for k in sorted(kv):
        if k.startswith(arch + ".") and any(
                s in k for s in ("block_count", "context_length",
                                 "embedding_length")):
            print(f"  {k:28s} {kv[k]}")
    types: dict[str, int] = {}
    for i in hdr["infos"]:
        types[_GGML.get(i["type"], (f"TYPE{i['type']}",))[0]] = \
            types.get(_GGML.get(i["type"], (f"TYPE{i['type']}",))[0], 0) + 1
    print(f"  tensor types                 {types}")

    if hdr.get("format") == "safetensors" and any(
            ".lora_A" in i["name"] or ".lora_B" in i["name"] for i in infos):
        return analyse_adapter(src, hdr, a.min_delta_pct
                               if a.min_delta_pct is not None else 1.0)
    if not a.ref:
        if not a.quiet:
            print("\n  (no --ref given; listing only)")
            for i in infos[:40]:
                print("    %-40s dims=%-22s %s" % (
                    i["name"], i["dims"],
                    _GGML.get(i["type"], (i["type"],))[0]))
            if len(infos) > 40:
                print(f"    ... {len(infos) - 40} more")
        print(f"\nbytes read: {src.fetched / 1e6:.1f} MB")
        return 0

    refs = [Ref(p) for p in a.ref[:2]]
    names = [os.path.basename(p.rstrip("/")) for p in a.ref[:2]]
    rows, counts = [], {"identical": 0, "convention": 0, "differs": 0,
                        "unverified": 0, "unmapped": 0, "quantised": 0}
    if hdr.get("format") == "safetensors" and any(
            ".lora_A" in i["name"] or ".lora_B" in i["name"] for i in infos):
        return analyse_adapter(src, hdr, a.min_delta_pct
                               if a.min_delta_pct is not None else 1.0)
    st = hdr.get("format") == "safetensors"
    for i in infos:
        cands, tf = ([i["name"]], "identity") if st else hf_candidates(i["name"], arch)
        key0 = refs[0].resolve(cands) if cands else None
        if key0 is None:
            counts["unmapped"] += 1
            rows.append({"tensor": i["name"], "verdict": "UNMAPPED"})
            continue
        g = read_tensor(src, hdr, i,
                        None if (a.full or i["type"] == 0) else a.sample_bytes)
        if g is None:
            counts["quantised"] += 1
            rows.append({"tensor": i["name"], "verdict": "QUANTISED"})
            continue
        ok0, how0, mx0 = _match(g, refs[0].get(key0), tf, a.full or i["type"] == 0)
        nparam = int(np.prod(i["dims"]))
        row = {"tensor": i["name"], "hf": key0, "transform": tf,
               "match_ref0": ok0, "how": how0, "max_abs": mx0,
               "params": nparam}
        if ok0:
            if how0 == "downcast":
                counts["convention"] += 1
                counts["downcast"] = counts.get("downcast", 0) + 1
                row["verdict"] = "PRECISION(ref F32 -> file BF16)"
            else:
                counts["identical" if tf == "identity" else "convention"] += 1
                row["verdict"] = ("IDENTICAL" if tf == "identity"
                                  else f"CONVENTION({how0})")
        else:
            # Two-reference disambiguation, applied BEFORE giving up on a
            # layout family: if ref0 and ref1 are bit-identical to each other
            # on this tensor but the GGUF differs from BOTH, only the converter
            # can explain it -- content cannot differ from two equal things.
            # This works on a byte PREFIX and needs no transform table, so it
            # settles the layout families cheaply too.
            verdict = None
            if len(refs) > 1:
                k1 = refs[1].resolve(cands)
                if k1 is not None:
                    t0, t1 = refs[0].get(key0), refs[1].get(k1)
                    refs_agree = (t0.shape == t1.shape and torch.equal(t0, t1))
                    ok1, _, _ = _match(g, t1, tf, a.full or i["type"] == 0)
                    row["refs_agree"] = bool(refs_agree)
                    row["match_ref1"] = bool(ok1)
                    if refs_agree and not ok1:
                        verdict = "CONVENTION(differs from both refs)"
                    elif ok1 and not ok0:
                        verdict = f"MATCHES {names[1]} NOT {names[0]}"
            if verdict is None and how0 == "unverified":
                counts["unverified"] += 1
                row["verdict"] = "UNVERIFIED(layout; use --full or a 2nd --ref)"
                rows.append(row)
                continue
            verdict = verdict or "DIFFERS"
            counts["convention" if verdict.startswith("CONVENTION")
                   else "differs"] += 1
            row["verdict"] = verdict
        rows.append(row)

    real = [r for r in rows if r["verdict"] == "DIFFERS"
            or r["verdict"].startswith("MATCHES")]
    compared = counts["identical"] + counts["convention"] + counts["differs"]
    # Weigh the verdict by PARAMETERS, not tensor count. 24 differing tensors
    # out of 307 is 7.8% and sounds decisive; those same 24 are 3,072 of 9.65 B
    # parameters = 0.000034%, which is the honest number. A tensor-count gate
    # would have waved through a model that is the base with 24 norms nudged.
    p_cmp = sum(r.get("params", 0) for r in rows
                if r["verdict"].startswith(("IDENTICAL", "CONVENTION",
                                            "DIFFERS", "MATCHES")))
    p_diff = sum(r.get("params", 0) for r in real)
    print(f"\nCOMPARED against ref0={names[0]}"
          + (f"  ref1={names[1]}" if len(refs) > 1 else ""))
    print(f"  IDENTICAL          : {counts['identical']}")
    print(f"  CONVENTION         : {counts['convention']}   "
          f"(converter transform, NOT a weight difference)")
    print(f"  CONTENT DIFFS      : {counts['differs']}   "
          f"(not converter artifacts -- see the params line below for whether "
          f"they AMOUNT to anything)")
    if counts.get("downcast"):
        print(f"    of which PRECISION-only : {counts['downcast']}  "
              f"(reference F32 stored as BF16 -- a re-save, NOT training)")
    if counts["unverified"]:
        print(f"  UNVERIFIED(layout) : {counts['unverified']}  (re-run --full)")
    if counts["unmapped"]:
        print(f"  UNMAPPED           : {counts['unmapped']}  "
              f"(no HF counterpart known for arch {arch!r})")
    if counts["quantised"]:
        print(f"  QUANTISED          : {counts['quantised']}  "
              f"(not elementwise-comparable)")
    if real and not a.quiet:
        print("\n  tensors differing for reasons the converter cannot explain:")
        for r in real[:60]:
            mx = r.get("max_abs")
            print("    %-36s %-40s max|d|=%s" % (
                r["tensor"], r["verdict"],
                "n/a" if mx is None or mx != mx else f"{mx:.6f}"))
        if len(real) > 60:
            print(f"    ... {len(real) - 60} more")
    print(f"\nbytes read: {src.fetched / 1e6:.1f} MB in {src.requests} requests")

    if a.json:
        with open(a.json, "w") as h:
            json.dump({"gguf": src.path or src.url, "arch": arch,
                       "kv": {k: v for k, v in kv.items()
                              if not isinstance(v, dict)},
                       "refs": a.ref[:2], "counts": counts, "rows": rows}, h,
                      indent=1)
        print(f"report: {a.json}")

    if compared:
        t_pct = 100.0 * counts["differs"] / compared
        p_pct = 100.0 * p_diff / p_cmp if p_cmp else 0.0
        print(f"\nreal difference: {counts['differs']}/{compared} tensors "
              f"({t_pct:.4f}%)  |  {p_diff:,}/{p_cmp:,} params ({p_pct:.6f}%)")
        if a.min_delta_pct is not None:
            print(f"gate: threshold {a.min_delta_pct}% of PARAMETERS")
            if p_pct < a.min_delta_pct:
                print("REFUSE: this GGUF is effectively the reference model -- "
                      "it carries no meaningful fine-tune signal.")
                return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        pass
