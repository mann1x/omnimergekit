#!/usr/bin/env python3
"""qwen_shared_upweight.py — Step 1b for Qwen3.5-MoE: dial up the always-on
shared expert to mask post-prune routing damage (rumination).

Qwen3.5-MoE has a parallel `mlp.shared_expert.{gate,up,down}_proj` + a sigmoid
`mlp.shared_expert_gate` at every MoE layer, running alongside the routed top-k
mixture: out = sigmoid(gate(x)) * shared_expert(x) + routed(x). Scaling the
shared_expert OUTPUT projection (down_proj) by alpha multiplies its contribution
by alpha — leaning harder on the reliable dense path when routing is unreliable.

bf16 model (NOT NVFP4A16), so we scale the raw down_proj.weight directly.
Main-model layers ONLY (`model.language_model.layers.*`); the MTP head's shared
expert is left untouched (drafter-only, doesn't affect verified output).

Reversible via .pre_shared_upweight backups. Writes a NEW output dir (keeps source).

Usage:
  python qwen_shared_upweight.py --src <bf16 dir> --out <new dir> --alpha 1.2
"""
import argparse, json, shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--target", default="mlp.shared_expert.down_proj.weight")
    ap.add_argument("--gate-scale", type=float, default=None,
                    help="Scale mlp.shared_expert_gate.weight by this (opens the sigmoid). None=off.")
    args=ap.parse_args()
    src=Path(args.src); out=Path(args.out)
    idx=json.load(open(src/"model.safetensors.index.json"))
    wm=idx["weight_map"]
    # main-model shared-expert down_proj only (exclude mtp.*)
    factor={}  # tensor -> scale factor
    for k in wm:
        if not k.startswith("model."): continue
        if k.endswith(args.target): factor[k]=args.alpha
        if args.gate_scale is not None and k.endswith("mlp.shared_expert_gate.weight"): factor[k]=args.gate_scale
    assert factor, "no target tensors"
    print(f"scaling {sum(1 for v in factor.values() if abs(v-args.alpha)<1e-9)} down_proj by {args.alpha}" + (f" + {sum(1 for v in factor.values() if args.gate_scale is not None and abs(v-args.gate_scale)<1e-9)} gate by {args.gate_scale}" if args.gate_scale is not None else ""))
    by_shard={}
    for t in factor: by_shard.setdefault(wm[t],[]).append(t)
    out.mkdir(parents=True, exist_ok=True)
    # copy all files, then rewrite edited shards
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, out/f.name)
    shards=sorted(set(wm.values()))
    for shard in shards:
        sd={}
        with safe_open(src/shard, framework="pt") as sf:
            meta=sf.metadata()
            for k in sf.keys():
                w=sf.get_tensor(k)
                if k in factor:
                    w=(w.float()*factor[k]).to(w.dtype)
                sd[k]=w
        save_file(sd, out/shard, metadata=meta)
        edited=len(by_shard.get(shard,[]))
        print(f"  {shard}: {edited} scaled" if edited else f"  {shard}: copied")
    print(f">>> DONE shared-upweight alpha={args.alpha} -> {out}")

if __name__=="__main__":
    main()
