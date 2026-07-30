#!/usr/bin/env python3
"""Probe a GGUF's KV metadata WITHOUT reading the tensor data.

Why this exists (and why it is not `gguf.GGUFReader`): GGUFReader mmaps the whole
file and builds the tensor table, so it needs the complete artifact. The two
questions we ask before adopting any third-party GGUF --

  1. are the EOG tokens right?   (eos=106 <turn|> for Gemma 4, NOT eos=1)
  2. is the chat template the current upstream one?

-- are answered entirely by the KV block at the very front of the file. For a 61 GB
BF16 that block is ~10-20 MB, so this reads a PREFIX and stops. Point it at an
`https://` URL and it pulls that prefix over HTTP Range requests: the verdict lands
before the download is committed, not after.

Usage
-----
  gguf_probe_header.py <path-or-url> [<path-or-url> ...]
        [--out-template DIR]   # dump each chat template to DIR/<name>.jinja
        [--tokens ID,ID,...]   # additionally resolve these raw ids to strings

Exit status is 0 on a clean parse; 2 if a file's metadata could not be read.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import struct
import sys

# GGUF metadata value types (gguf/constants.py GGUFValueType)
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)

_FIXED = {
    U8: ("<B", 1), I8: ("<b", 1),
    U16: ("<H", 2), I16: ("<h", 2),
    U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4),
    BOOL: ("<?", 1),
    U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8),
}

# Keys whose values we keep. Everything else is parsed-and-dropped so a 262k-entry
# token array never costs us more than one pass.
WANTED_SCALARS = (
    "general.architecture", "general.name", "general.file_type",
    "general.size_label", "general.basename", "general.quantization_version",
    "tokenizer.ggml.model", "tokenizer.ggml.pre",
    "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id",
    "tokenizer.ggml.unknown_token_id", "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.separator_token_id",
    "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token",
    "tokenizer.chat_template",
)
# Arrays we retain in full (needed to resolve ids -> strings).
WANTED_ARRAYS = ("tokenizer.ggml.tokens", "tokenizer.ggml.eog_token_ids")

# Every id-valued key above, for the resolve table.
ID_KEYS = tuple(k for k in WANTED_SCALARS if k.endswith("_token_id"))


class PrefixReader:
    """Sequential byte source over a local file or an HTTP range-served URL."""

    CHUNK = 8 << 20

    def __init__(self, src: str):
        self.src = src
        self.pos = 0
        self._http = src.startswith("http://") or src.startswith("https://")
        if self._http:
            import requests  # local import: only the URL path needs it
            self._sess = requests.Session()
            self._buf = bytearray()
            self._base = 0  # file offset of _buf[0]
        else:
            self._fh = open(src, "rb")

    def read(self, n: int) -> bytes:
        if not self._http:
            b = self._fh.read(n)
            if len(b) < n:
                raise EOFError(f"short read at {self.pos} (+{n})")
            self.pos += n
            return b
        while self.pos + n > self._base + len(self._buf):
            self._fetch()
        off = self.pos - self._base
        b = bytes(self._buf[off:off + n])
        self.pos += n
        # Release consumed prefix so memory stays flat over a long token array.
        if off > self.CHUNK:
            del self._buf[:off]
            self._base += off
        return b

    def _fetch(self) -> None:
        start = self._base + len(self._buf)
        end = start + self.CHUNK - 1
        r = self._sess.get(self.src, headers={"Range": f"bytes={start}-{end}"},
                           timeout=120, stream=True)
        if r.status_code not in (200, 206):
            raise IOError(f"HTTP {r.status_code} fetching bytes {start}-{end}")
        body = r.content
        if not body:
            raise EOFError(f"empty range response at {start}")
        self._buf.extend(body)

    def skip(self, n: int) -> None:
        """Advance without materializing bytes (cheap for local files)."""
        if not self._http:
            self._fh.seek(n, io.SEEK_CUR)
            self.pos += n
        else:
            self.read(n)

    def close(self) -> None:
        if not self._http:
            self._fh.close()


def _scalar(r: PrefixReader, t: int):
    if t in _FIXED:
        fmt, size = _FIXED[t]
        return struct.unpack(fmt, r.read(size))[0]
    if t == STRING:
        (n,) = struct.unpack("<Q", r.read(8))
        return r.read(n).decode("utf-8", "replace")
    raise ValueError(f"unhandled value type {t}")


def _array(r: PrefixReader, keep: bool):
    (et,) = struct.unpack("<I", r.read(4))
    (count,) = struct.unpack("<Q", r.read(8))
    if et == STRING:
        out = [] if keep else None
        for _ in range(count):
            (n,) = struct.unpack("<Q", r.read(8))
            b = r.read(n)
            if keep:
                out.append(b.decode("utf-8", "replace"))
        return out
    if et == ARRAY:
        return [_array(r, keep) for _ in range(count)]
    fmt, size = _FIXED[et]
    if not keep:
        r.skip(size * count)          # bulk-skip scores / token_type
        return None
    raw = r.read(size * count)
    return list(struct.unpack("<" + fmt[1] * count, raw))


def probe(src: str) -> dict:
    r = PrefixReader(src)
    try:
        magic = r.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF (magic={magic!r})")
        version, n_tensors, n_kv = struct.unpack("<IQQ", r.read(20))
        got: dict = {"_version": version, "_n_tensors": n_tensors, "_n_kv": n_kv}
        for _ in range(n_kv):
            (klen,) = struct.unpack("<Q", r.read(8))
            key = r.read(klen).decode("utf-8", "replace")
            (vt,) = struct.unpack("<I", r.read(4))
            if vt == ARRAY:
                v = _array(r, keep=key in WANTED_ARRAYS)
                if v is not None:
                    got[key] = v
            else:
                v = _scalar(r, vt)
                if key in WANTED_SCALARS:
                    got[key] = v
        got["_kv_bytes"] = r.pos
        return got
    finally:
        r.close()


def report(src: str, got: dict, out_dir: str | None, extra_ids: list[int]) -> None:
    name = os.path.basename(src.split("?")[0])
    toks = got.get("tokenizer.ggml.tokens") or []

    def tok(i):
        if i is None or not isinstance(i, int):
            return "-"
        return repr(toks[i]) if 0 <= i < len(toks) else "<out-of-range>"

    print(f"FILE {name}")
    print(f"  src                  = {src}")
    print(f"  gguf version         = {got['_version']}   tensors={got['_n_tensors']}  "
          f"kv={got['_n_kv']}  kv_bytes={got['_kv_bytes']:,}")
    for k in ("general.architecture", "general.name", "general.size_label",
              "general.file_type", "tokenizer.ggml.model", "tokenizer.ggml.pre"):
        if k in got:
            print(f"  {k:<20} = {got[k]}")
    print(f"  vocab size           = {len(toks):,}")

    print("  --- EOG / special tokens ---")
    for k in ID_KEYS:
        if k in got:
            short = k.replace("tokenizer.ggml.", "")
            print(f"  {short:<20} = {got[k]:<8} {tok(got[k])}")
    for k in ("tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token"):
        if k in got:
            print(f"  {k.replace('tokenizer.ggml.', ''):<20} = {got[k]}")
    if "tokenizer.ggml.eog_token_ids" in got:
        ids = got["tokenizer.ggml.eog_token_ids"]
        print(f"  eog_token_ids        = {ids} -> {[tok(i) for i in ids]}")
    for i in extra_ids:
        print(f"  [id {i}]              = {tok(i)}")

    ct = got.get("tokenizer.chat_template")
    print("  --- chat template ---")
    if not ct:
        print("  chat_template        = <ABSENT>  (llama-server --jinja has nothing to load)")
    else:
        h = hashlib.sha256(ct.encode()).hexdigest()
        print(f"  chat_template        = {len(ct):,} chars  sha256={h[:32]}")
        print(f"  head                 = {ct[:160]!r}")
        print(f"  tail                 = {ct[-160:]!r}")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            dst = os.path.join(out_dir, name.replace(".gguf", "") + ".jinja")
            with open(dst, "w") as fh:
                fh.write(ct)
            print(f"  dumped               = {dst}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("srcs", nargs="+", help="local .gguf paths and/or https URLs")
    ap.add_argument("--out-template", metavar="DIR",
                    help="dump each chat template to DIR/<name>.jinja for diffing")
    ap.add_argument("--tokens", default="",
                    help="comma-separated extra token ids to resolve (e.g. 1,105,106,107)")
    a = ap.parse_args()
    extra = [int(x) for x in a.tokens.split(",") if x.strip()]

    rc = 0
    for src in a.srcs:
        try:
            report(src, probe(src), a.out_template, extra)
        except Exception as e:              # noqa: BLE001 - report and keep going
            print(f"FILE {os.path.basename(src)}\n  !! PROBE FAILED: "
                  f"{type(e).__name__}: {e}\n", file=sys.stderr)
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
