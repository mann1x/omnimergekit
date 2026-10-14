#!/usr/bin/env python3
"""gguf_read_prism_meta.py — dump a GGUF's KV section WITHOUT gguf-py.

WHY. gguf-py's GGUFReader validates every tensor against GGMLQuantizationType and
raises `ValueError: 142 is not a valid GGMLQuantizationType` before any KV is
reachable, so it cannot read a PrismML Bonsai / PQ2_0 / PTQ1_0 file at all. This
walks the KV section only and never looks at the tensor table, so it answers the
same way for known and unknown weight types. Same technique as
quantize_gguf.py:_gguf_file_type, generalised to every key.

Usage: gguf_read_prism_meta.py FILE [key-substring ...]
"""
from __future__ import annotations

import struct
import sys

W = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
     5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8),
     12: ("d", 8)}
T_STR, T_ARR = 8, 9


def read(path):
    f = open(path, "rb")
    if f.read(4) != b"GGUF":
        sys.exit("not a GGUF")
    ver, = struct.unpack("<I", f.read(4))
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

    def rd_str():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def rd_val(t):
        if t in W:
            fmt, sz = W[t]
            return struct.unpack("<" + fmt, f.read(sz))[0]
        if t == T_STR:
            return rd_str()
        if t == T_ARR:
            et, = struct.unpack("<I", f.read(4))
            cnt, = struct.unpack("<Q", f.read(8))
            if et in W:
                fmt, sz = W[et]
                return list(struct.unpack("<%d%s" % (cnt, fmt), f.read(sz * cnt)))
            if et == T_STR:
                return [rd_str() for _ in range(cnt)]
            sys.exit("nested array type %d" % et)
        sys.exit("unknown KV type %d" % t)

    out = {}
    for _ in range(n_kv):
        k = rd_str()
        t, = struct.unpack("<I", f.read(4))
        out[k] = rd_val(t)
    return ver, n_tensors, n_kv, out


GGML_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
              8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
              14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
              19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
              24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
              30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
              142: "PQ2_0", 143: "PTQ1_0"}


def read_tensors(path):
    """Tensor table: (name, dims, ggml_type). Reads the type as a plain uint32 and
    NEVER validates it against an enum, which is the whole reason this works on a
    file holding PQ2_0 / PTQ1_0 (142 / 143)."""
    f = open(path, "rb")
    if f.read(4) != b"GGUF":
        sys.exit("not a GGUF")
    struct.unpack("<I", f.read(4))
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

    def rd_str():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def skip_val(t):
        if t in W:
            f.seek(W[t][1], 1)
        elif t == T_STR:
            rd_str()
        elif t == T_ARR:
            et, = struct.unpack("<I", f.read(4))
            cnt, = struct.unpack("<Q", f.read(8))
            if et in W:
                f.seek(W[et][1] * cnt, 1)
            elif et == T_STR:
                for _ in range(cnt):
                    rd_str()
            else:
                sys.exit("nested array %d" % et)
        else:
            sys.exit("unknown KV type %d" % t)

    for _ in range(n_kv):
        rd_str()
        t, = struct.unpack("<I", f.read(4))
        skip_val(t)

    out = []
    for _ in range(n_tensors):
        name = rd_str()
        nd, = struct.unpack("<I", f.read(4))
        dims = list(struct.unpack("<%dQ" % nd, f.read(8 * nd)))
        ttype, = struct.unpack("<I", f.read(4))
        struct.unpack("<Q", f.read(8))          # offset
        out.append((name, dims, ttype))
    return out


def main():
    if "--tensors" in sys.argv:
        path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
        ts = read_tensors(path)
        from collections import Counter
        c = Counter(GGML_NAMES.get(t, "type%d" % t) for _, _, t in ts)
        print("tensors=%d  type census: %s" % (len(ts), dict(c)))
        for name, dims, t in ts:
            print("  %-44s %-18s %s" % (name, GGML_NAMES.get(t, "type%d" % t), dims))
        return
    path = sys.argv[1]
    pats = [p.lower() for p in sys.argv[2:]] or ["prism"]
    ver, nt, nkv, kv = read(path)
    print("gguf v%d  tensors=%d  kv=%d  %s" % (ver, nt, nkv, path))
    hits = {k: v for k, v in kv.items() if any(p in k.lower() for p in pats)}
    if not hits:
        print("  (no key matching %s)" % pats)
        print("  all keys:")
        for k in sorted(kv):
            print("    %s" % k)
        return
    for k in sorted(hits):
        v = hits[k]
        if isinstance(v, list):
            uniq = sorted(set(v)) if len(v) and not isinstance(v[0], str) else None
            print("  %-42s list[%d]" % (k, len(v)))
            print("      first 24 : %s" % (v[:24],))
            if uniq is not None and len(uniq) <= 8:
                from collections import Counter
                c = Counter(v)
                print("      distinct : %s   counts=%s" % (uniq, dict(c)))
        else:
            print("  %-42s %s" % (k, str(v)[:160]))


if __name__ == "__main__":
    main()
