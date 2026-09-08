#!/usr/bin/env python
"""Graft the published 184e MTP block into every REAM arm, in place.

WHY THIS AND NOT A MERGE
------------------------
Qwen3.6-35B-A3B carries an MTP (next-token-prediction) block that convert_hf_to_gguf
counts as a 41st block: config has mtp_num_hidden_layers=1, so the GGUF gets
block_count=41. REAM's save_pretrained drops mtp.* unless --mtp-safetensors is passed,
which our arm builds never did -- so every arm produced a GGUF declaring 41 blocks while
shipping 40. llama-quantize builds no graph and did not care; llama-imatrix does, and
died with 'missing tensor blk.40.attn_norm.weight'. Every arm was unservable.

Passing --mtp-safetensors is NOT the fix here. REAM's build_mtp_layer_qwen3_5 assembles a
Qwen3MoeDecoderLayer + Qwen3MoeSparseMoeBlock -- the OLD Qwen3-MoE shapes: unpacked
per-expert gate/up/down and no shared expert. Qwen3.6's MTP block is packed
(experts.gate_up_proj [E,1024,2048]) and HAS mlp.shared_expert.* + shared_expert_gate.
Its _load_weights PRINTS missing/mismatched keys and then continues, so that path would
have loaded a structurally wrong drafter without failing.

The block is inert for this experiment:
  * it is a speculative-decode drafter; llama.cpp skips it in the forward graph,
  * eval_ream_arms.sh deliberately never sets LLAMA_ARG_SPEC_TYPE,
  * and the PUBLISHED anchor's own MTP is pruned-but-not-merged.
So the right move is to hold it CONSTANT and equal to the anchor's, exactly as we forced
the imatrix recipe to match the anchor's. A component that varies per arm while
contributing nothing to the measurement is pure noise in the provenance; a component
held identical across arms cannot explain any difference between them.

Every arm therefore gets the published 184e mtp.* verbatim. Documented, uniform, inert.
"""
import json
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/srv/ml/models/Qwen3.6-35B-A3B-184e-coder-lcbmpe"
WORK = "/mnt/sdc/ream-work"
ARMS = sys.argv[1:] or ["armD_ourssal_nomerge", "armC", "armB", "armE",
                        "armF_rnorm_nomerge"]
SHARD = "mtp.safetensors"


def load_src_mtp():
    idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    keys = sorted(k for k in idx if k.startswith("mtp."))
    if len(keys) != 19:
        raise SystemExit(f"FAIL: source has {len(keys)} mtp keys, expected 19")
    out = {}
    for k in keys:
        with safe_open(os.path.join(SRC, idx[k]), framework="pt") as f:
            out[k] = f.get_tensor(k)
    return out


def graft(arm, mtp):
    d = os.path.join(WORK, arm)
    ipath = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(ipath):
        print(f"  SKIP {arm}: no index at {ipath}")
        return False
    idx = json.load(open(ipath))
    have = [k for k in idx["weight_map"] if k.startswith("mtp.")]
    if len(have) == 19:
        print(f"  SKIP {arm}: already carries 19 mtp keys")
        return True

    save_file(mtp, os.path.join(d, SHARD), metadata={"format": "pt"})
    for k in mtp:
        idx["weight_map"][k] = SHARD
    json.dump(idx, open(ipath, "w"), indent=2)

    # ARTIFACT gate: re-read from disk. Writing proves intent, reading proves the loader
    # will find it -- the same distinction that let the broken arms pass for a whole day.
    back = json.load(open(ipath))["weight_map"]
    bad = [k for k in mtp if back.get(k) != SHARD]
    if bad:
        raise SystemExit(f"FAIL {arm}: {len(bad)} keys absent from index, e.g. {bad[:3]}")
    with safe_open(os.path.join(d, SHARD), framework="pt") as f:
        n = len(f.keys())
    sz = os.path.getsize(os.path.join(d, SHARD)) / 1e9
    print(f"  OK {arm}: {n} mtp tensors, {sz:.2f} GB -> {SHARD}")
    return True


def main():
    mtp = load_src_mtp()
    tot = sum(v.numel() * v.element_size() for v in mtp.values()) / 1e9
    print(f"source MTP block: {len(mtp)} tensors, {tot:.2f} GB (from {SRC})")
    print(f"  experts: {tuple(mtp['mtp.layers.0.mlp.experts.gate_up_proj'].shape)}")
    ok = sum(graft(a, mtp) for a in ARMS)
    print(f">>> MTP_GRAFT_DONE ok={ok}/{len(ARMS)}")
    sys.exit(0 if ok == len(ARMS) else 1)


if __name__ == "__main__":
    main()
