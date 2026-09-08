#!/usr/bin/env python
"""Recover each arm's KEEP SET from its weights and diff it against another arm's.

Neither arm dir stores keep metadata, and the REAM build log only prints layer 0's grouping,
so the authoritative source is the weights themselves. Both no-merge arms carry surviving
experts VERBATIM from the base (no averaging), so an exact byte match on a slice of
down_proj recovers each kept expert's ORIGINAL index. Base stores experts stacked
[256, 2048, 512]; arms store them split per expert at [2048, 512] -- same layout, so the
slices are directly comparable with no transpose ambiguity.

8 rows x 512 cols of bf16 = 8 KB/expert fingerprint; collision between two distinct experts
on 4096 exact bf16 values does not happen. Verified by requiring a UNIQUE match per expert.
"""
import json
import sys
from collections import defaultdict

import torch
from safetensors import safe_open

BASE = "/srv/ml/models/Qwen3.6-35B-A3B"
ROWS = 8


def layers_of(d):
    ix = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
    ls = set()
    for k in ix:
        if k.startswith("model.language_model.layers.") and ".mlp.experts." in k:
            ls.add(int(k.split("layers.")[1].split(".")[0]))
    return ix, sorted(ls)


def base_fp(ix, layer):
    """original_expert_index -> fingerprint bytes"""
    k = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
    with safe_open(f"{BASE}/{ix[k]}", framework="pt") as f:
        sl = f.get_slice(k)
        n = sl.get_shape()[0]
        return {e: sl[e, :ROWS, :].contiguous().view(torch.int16).numpy().tobytes()
                for e in range(n)}


def arm_keep(d, ix, layer, table):
    """Recover the arm's keep set for one layer as a set of ORIGINAL base indices."""
    keep, unmatched = [], 0
    slot = 0
    while True:
        k = f"model.language_model.layers.{layer}.mlp.experts.{slot}.down_proj.weight"
        if k not in ix:
            break
        with safe_open(f"{d}/{ix[k]}", framework="pt") as f:
            fp = f.get_slice(k)[:ROWS, :].contiguous().view(torch.int16).numpy().tobytes()
        hits = [e for e, b in table.items() if b == fp]
        if len(hits) == 1:
            keep.append(hits[0])
        else:
            unmatched += 1
        slot += 1
    return keep, slot, unmatched


A, B = sys.argv[1], sys.argv[2]
ixa, la = layers_of(A)
ixb, lb = layers_of(B)
layers = sorted(set(la) & set(lb))
print(f"A = {A}\nB = {B}\nlayers = {len(layers)}  (rows fingerprinted: {ROWS})\n")

ixbase = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
tot_a = tot_b = tot_common = tot_unmatched = 0
per_layer = []
churn = defaultdict(int)
for L in layers:
    table = base_fp(ixbase, L)
    ka, na, ua = arm_keep(A, ixa, L, table)
    kb, nb, ub = arm_keep(B, ixb, L, table)
    sa, sb = set(ka), set(kb)
    per_layer.append((L, len(sa), len(sb), len(sa & sb), len(sa - sb)))
    tot_a += len(sa); tot_b += len(sb); tot_common += len(sa & sb)
    tot_unmatched += ua + ub
    churn[len(sa - sb)] += 1

print(f"{'layer':>5} {'|A|':>5} {'|B|':>5} {'shared':>7} {'A-only':>7} {'%diff':>7}")
for L, a, b, c, d in per_layer:
    print(f"{L:>5} {a:>5} {b:>5} {c:>7} {d:>7} {100*d/max(a,1):>6.1f}%")

n = len(layers)
print(f"\nTOTAL kept: A={tot_a}  B={tot_b}  shared={tot_common}  "
      f"A-only={tot_a - tot_common}  ({100*(tot_a-tot_common)/max(tot_a,1):.1f}% of A's keeps)")
print(f"mean per-layer disagreement: {(tot_a-tot_common)/n:.1f} experts")
print(f"per-layer A-only histogram: {dict(sorted(churn.items()))}")
if tot_unmatched:
    print(f"\nWARNING: {tot_unmatched} expert(s) had no unique base match -- NOT verbatim "
          f"survivors (merged/averaged?). Keep-set recovery is incomplete for those.")
else:
    print("\nAll experts in both arms matched a base expert byte-exactly (verbatim survivors).")
