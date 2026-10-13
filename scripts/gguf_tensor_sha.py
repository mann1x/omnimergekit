#!/usr/bin/env python3
"""gguf_tensor_sha.py FILE PREFIX — sha256 of each tensor's raw bytes whose name starts with PREFIX.

Manual walker (gguf-py cannot open PQ2_0/PTQ1_0 files). A tensor's byte span is taken from the
offset table itself (next offset, or EOF), so no per-type size table is needed and an unknown
type cannot be mis-sized. Prints `name type dims sha16 nbytes`, one per line, name-sorted.
"""
import hashlib
import os
import struct
import sys

W = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def main():
    path, prefix = sys.argv[1], sys.argv[2]
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
            name = rstr()
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            typ, off = struct.unpack("<IQ", f.read(12))
            ti.append((name, typ, dims, off))
        base = (f.tell() + align - 1) // align * align
        end = os.path.getsize(path) - base
        offs = sorted({o for _, _, _, o in ti}) + [end]
        nxt = {o: offs[i + 1] for i, o in enumerate(offs[:-1])}
        for name, typ, dims, off in sorted(ti):
            if not name.startswith(prefix):
                continue
            n = 1
            for d in dims:
                n *= d
            # exact size for the full-precision types the head uses; span otherwise
            size = {0: 4 * n, 1: 2 * n, 30: 2 * n}.get(typ, nxt[off] - off)
            f.seek(base + off)
            h = hashlib.sha256()
            left = size
            while left:
                b = f.read(min(left, 1 << 24))
                h.update(b)
                left -= len(b)
            print(name, typ, list(dims), h.hexdigest()[:16], size)


main()
