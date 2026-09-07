#!/usr/bin/env python
"""Recover an arm's per-layer keep set from its WEIGHTS and save it as JSON.

No-merge arms carry surviving experts verbatim from the base, so an exact byte match on a
slice of down_proj recovers each kept expert's original index. Base stores experts stacked
[256, 2048, 512]; arms store them split per expert at [2048, 512] -- same layout, no
transpose ambiguity. 8 rows x 512 bf16 = 8 KB/expert; a collision on 4096 exact bf16 values
does not happen, and a UNIQUE match is required per expert regardless.

This is the ground truth the REAP-saliency dump is gated against: if the re-profiled
saliency's top-k does not reproduce armE's recovered keep set, the dump is not the vector
armE actually selected with and must not be used to build anything.

  python recover_keepsets.py out.json armE /mnt/sdc/ream-work/armE [name dir ...]
"""
import json
import os
import sys

import torch
from safetensors import safe_open

BASE = os.environ.get("REAP_BASE_MODEL", "/srv/ml/models/Qwen3.6-35B-A3B")
ROWS = 8


def _index(d):
    return json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]


def _layers(ix):
    ls = set()
    for k in ix:
        if k.startswith("model.language_model.layers.") and ".mlp.experts." in k:
            ls.add(int(k.split("layers.")[1].split(".")[0]))
    return sorted(ls)


def _fp(t):
    return t.contiguous().view(torch.int16).numpy().tobytes()


def base_table(ixbase, layer):
    k = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
    with safe_open(f"{BASE}/{ixbase[k]}", framework="pt") as f:
        sl = f.get_slice(k)
        return {e: _fp(sl[e, :ROWS, :]) for e in range(sl.get_shape()[0])}


def _match(fp, table, keep, unmatched):
    hits = [e for e, b in table.items() if b == fp]
    if len(hits) == 1:
        keep.append(hits[0])
        return unmatched
    return unmatched + 1


def keepset(d, ix, layer, table):
    """Recover a layer's keep set, supporting BOTH arm expert layouts.

    REAM-built arms store experts SPLIT per expert (`experts.{i}.down_proj.weight`).
    expert_drop-built arms (Ornith Arm A) slice the base tensor and keep it STACKED
    (`experts.down_proj`, [E, 2048, 512]) -- identical in form to the base. The original
    code only handled the split case and, on a stacked arm, broke at slot 0 and returned
    an EMPTY keep set with unmatched=0, i.e. a silent wrong answer that passed its own
    gate. Both layouts are now handled explicitly and an unknown layout is fatal.
    """
    keep, unmatched = [], 0
    split0 = f"model.language_model.layers.{layer}.mlp.experts.0.down_proj.weight"
    stacked = f"model.language_model.layers.{layer}.mlp.experts.down_proj"

    if split0 in ix:
        slot = 0
        while True:
            k = f"model.language_model.layers.{layer}.mlp.experts.{slot}.down_proj.weight"
            if k not in ix:
                break
            with safe_open(f"{d}/{ix[k]}", framework="pt") as f:
                fp = _fp(f.get_slice(k)[:ROWS, :])
            unmatched = _match(fp, table, keep, unmatched)
            slot += 1
        return sorted(keep), slot, unmatched

    if stacked in ix:
        with safe_open(f"{d}/{ix[stacked]}", framework="pt") as f:
            sl = f.get_slice(stacked)
            n = sl.get_shape()[0]
            for slot in range(n):
                unmatched = _match(_fp(sl[slot, :ROWS, :]), table, keep, unmatched)
        return sorted(keep), n, unmatched

    sys.exit(f"ABORT: layer {layer} has neither split nor stacked expert tensors")


def main():
    out_path, rest = sys.argv[1], sys.argv[2:]
    if not rest or len(rest) % 2:
        sys.exit("usage: recover_keepsets.py OUT.json NAME DIR [NAME DIR ...]")
    arms = list(zip(rest[::2], rest[1::2]))

    ixbase = _index(BASE)
    result, bad = {}, 0
    for name, d in arms:
        ix = _index(d)
        layers = _layers(ix)
        per = {}
        for L in layers:
            table = base_table(ixbase, L)
            keep, n, un = keepset(d, ix, L, table)
            per[str(L)] = keep
            bad += un
            if un:
                print(f"  {name} L{L}: {un}/{n} experts had NO unique base match "
                      f"(not a verbatim survivor)")
        result[name] = per
        sizes = sorted({len(v) for v in per.values()})
        print(f"{name}: layers={len(per)} keep_per_layer={sizes}")
        empty = [L for L, v in per.items() if not v]
        if empty:
            sys.exit(f"REFUSING to write: {name} recovered ZERO keeps on {len(empty)} "
                     f"layer(s) (e.g. {empty[:5]}). An empty keep set is a failed "
                     f"recovery, not an empty arm.")

    if bad:
        sys.exit(f"REFUSING to write: {bad} expert(s) unmatched -- these arms are not pure "
                 f"verbatim-survivor cuts, so a 'keep set' is not well defined for them.")

    with open(out_path, "w") as fh:
        json.dump(result, fh)
    print(f">>> KEEPSETS_OK arms={list(result)} -> {out_path}")


if __name__ == "__main__":
    main()
