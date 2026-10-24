#!/usr/bin/env python3
"""Strip the NextN / MTP draft block(s) from a GGUF, leaving a plain target model.

Use case: publish a quant whose speculative drafter is an EXTERNAL model (e.g. a DFlash
drafter attached with ollama `DRAFT`), so the built-in MTP head is dead weight.

What changes, and nothing else:
  - tensors of the last `<arch>.nextn_predict_layers` blocks (blk.{B-N}..blk.{B-1}) are dropped;
  - `<arch>.block_count` B -> B-N;
  - `<arch>.nextn_predict_layers` is removed (a plain conversion writes no such key);
  - every per-layer ARRAY KV of length B is truncated to its first B-N entries
    (qwen35: `attention.recurrent_layers`).
Every kept tensor is copied byte for byte with its original ggml type, so quant types and the
imatrix effect are untouched; `--verify` re-reads both files and proves it.

Refuses (exit 2) when the GGUF has no nextn layers, when a tensor name outside the stripped
blocks mentions nextn/mtp, or when a per-layer array has an unexpected length.

  gguf_strip_nextn.py IN.gguf OUT.gguf [--verify]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

import gguf


def die(msg: str) -> None:
    print(f"STRIP_NEXTN_REFUSED: {msg}", file=sys.stderr)
    sys.exit(2)


def field_value(field):
    return field.contents()


def strip(src: str, dst: str) -> dict:
    r = gguf.GGUFReader(src)
    arch = field_value(r.fields[gguf.Keys.General.ARCHITECTURE])
    k_blocks, k_nextn = f"{arch}.block_count", f"{arch}.nextn_predict_layers"
    if k_blocks not in r.fields:
        die(f"no {k_blocks}")
    B = int(field_value(r.fields[k_blocks]))
    N = int(field_value(r.fields[k_nextn])) if k_nextn in r.fields else 0
    if N <= 0:
        die(f"{k_nextn} is {N}: nothing to strip")
    keep_blocks = B - N
    blk_re = re.compile(r"^blk\.(\d+)\.")

    dropped, kept = [], []
    for t in r.tensors:
        m = blk_re.match(t.name)
        if m and int(m.group(1)) >= keep_blocks:
            dropped.append(t)
        else:
            if re.search(r"nextn|(^|\.)mtp\.", t.name):
                die(f"nextn/mtp tensor outside the stripped blocks: {t.name}")
            kept.append(t)
    if not dropped:
        die(f"no tensors in blocks >= {keep_blocks}")

    w = gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
    al = r.fields.get(gguf.Keys.General.ALIGNMENT)
    if al is not None:
        w.data_alignment = int(field_value(al))

    truncated = []
    for f in r.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF.") or f.name == k_nextn:
            continue
        vt = f.types[0]
        sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        val = field_value(f)
        if f.name == k_blocks:
            val = keep_blocks
        elif vt == gguf.GGUFValueType.ARRAY and f.name.startswith(f"{arch}.") and len(f.data) == B:
            val = list(val)[:keep_blocks]
            truncated.append(f.name)
        elif (vt == gguf.GGUFValueType.ARRAY and f.name.startswith(f"{arch}.")
              and len(f.data) not in (B, keep_blocks) and "layer" in f.name):
            die(f"per-layer array {f.name} has length {len(f.data)}, block_count {B}")
        w.add_key_value(f.name, val, vt, sub_type=sub)

    for t in kept:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in kept:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    return {"arch": arch, "block_count": [B, keep_blocks], "dropped": len(dropped),
            "kept": len(kept), "truncated": truncated}


def verify(src: str, dst: str, info: dict) -> None:
    a, b = gguf.GGUFReader(src), gguf.GGUFReader(dst)
    ta = {t.name: t for t in a.tensors}
    tb = {t.name: t for t in b.tensors}
    keep = info["block_count"][1]
    want = {n for n in ta if not (re.match(r"^blk\.(\d+)\.", n) and int(n.split(".")[1]) >= keep)}
    if set(tb) != want:
        die(f"tensor set mismatch: missing {sorted(want - set(tb))[:5]} extra {sorted(set(tb) - want)[:5]}")
    for n in want:
        x, y = ta[n], tb[n]
        if x.tensor_type != y.tensor_type or list(x.shape) != list(y.shape):
            die(f"{n}: type/shape changed")
        if hashlib.sha256(x.data.tobytes()).digest() != hashlib.sha256(y.data.tobytes()).digest():
            die(f"{n}: bytes changed")
    arch = info["arch"]
    if f"{arch}.nextn_predict_layers" in b.fields:
        die("nextn_predict_layers still present")
    if int(field_value(b.fields[f"{arch}.block_count"])) != keep:
        die("block_count not updated")
    for k in ("general.name", "tokenizer.ggml.eos_token_id", "quantize.imatrix.dataset", "general.file_type"):
        fa, fb = a.fields.get(k), b.fields.get(k)
        if (fa is None) != (fb is None) or (fa is not None and field_value(fa) != field_value(fb)):
            die(f"KV {k} changed")
    print(f"    verify: {len(want)} tensors byte-identical, block_count {keep}, no nextn key")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--verify", action="store_true", help="re-read both files and prove every kept tensor is identical")
    args = ap.parse_args()
    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        die("src and dst are the same file")
    part = args.dst + ".part"
    info = strip(args.src, part)
    if args.verify:
        verify(args.src, part, info)
    os.replace(part, args.dst)
    print(f"STRIP_NEXTN_OK {os.path.basename(args.dst)} arch={info['arch']} block_count "
          f"{info['block_count'][0]}->{info['block_count'][1]} dropped={info['dropped']} kept={info['kept']} "
          f"truncated={info['truncated']} size={os.path.getsize(args.dst)}")


if __name__ == "__main__":
    main()
