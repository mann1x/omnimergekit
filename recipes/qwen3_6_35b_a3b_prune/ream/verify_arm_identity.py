#!/usr/bin/env python
"""Prove a REAM-built arm keeps EXACTLY the experts our shipped drop map keeps.

This is the arm-D integration gate. If it does not pass, nothing produced by the REAM
path is believable, because we would not know whose experts we are measuring.

Method: recover the surviving expert identities from the WEIGHTS, not from a log. For each
layer we take the built model's router rows (mlp.gate.weight, shape [184, 2048]) and match
each one against the base model's 256 rows. Router rows are high-dimensional and distinct,
so an exact match is an identity proof, and it does not depend on the builder having
recorded anything. We then compare that recovered keep set against the drop map.

Also compares against the published 184e model when given, tensor-by-tensor -- with
--merging none the expert weights are sliced, not modified, so they must be BIT-IDENTICAL.
Any difference means the merge path touched weights it should not have.
"""
import argparse
import json
import sys

import torch
from safetensors import safe_open


def load_layer_tensors(model_dir, names):
    """-> {name: tensor} pulled from the sharded safetensors without loading the model."""
    with open(f"{model_dir}/model.safetensors.index.json") as fh:
        index = json.load(fh)["weight_map"]
    want = {}
    for n in names:
        if n not in index:
            raise SystemExit(f"refusing: {n} not in {model_dir} index")
        want.setdefault(index[n], []).append(n)
    out = {}
    for shard, keys in want.items():
        with safe_open(f"{model_dir}/{shard}", framework="pt") as f:
            for k in keys:
                out[k] = f.get_tensor(k)
    return out


def router_name(li):
    return f"model.language_model.layers.{li}.mlp.gate.weight"


def _index_of(model_dir):
    with open(f"{model_dir}/model.safetensors.index.json") as fh:
        return json.load(fh)["weight_map"]


def expert_weight(model_dir, li, e, which):
    """One expert's [out, in] weight, whichever layout the checkpoint uses.

    The base/published Qwen3.6 checkpoints store experts PACKED
    (mlp.experts.gate_up_proj [E, 2*n_ff, n_embd], mlp.experts.down_proj [E, n_embd, n_ff]),
    but a model written by REAM's save_pretrained comes back UNPACKED as a ModuleList
    (mlp.experts.{e}.{gate,up,down}_proj.weight). Comparing the two therefore needs a
    normaliser, not a single name -- which is why the first version of this check found zero
    tensors to compare and silently had nothing to say.
    """
    base = f"model.language_model.layers.{li}.mlp.experts"
    index = _index_of(model_dir)

    unpacked = f"{base}.{e}.{which}.weight"
    if unpacked in index:
        return load_layer_tensors(model_dir, [unpacked])[unpacked]

    if which == "down_proj":
        n = f"{base}.down_proj"
        n = n if n in index else n + ".weight"
        if n not in index:
            raise SystemExit(f"refusing: no down_proj for L{li} in {model_dir}")
        return load_layer_tensors(model_dir, [n])[n][e]

    n = f"{base}.gate_up_proj"
    n = n if n in index else n + ".weight"
    if n not in index:
        raise SystemExit(f"refusing: no gate_up_proj for L{li} in {model_dir}")
    packed = load_layer_tensors(model_dir, [n])[n][e]        # [2*n_ff, n_embd]
    n_ff = packed.shape[0] // 2
    return packed[:n_ff] if which == "gate_proj" else packed[n_ff:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="256e base model dir")
    ap.add_argument("--built", required=True, help="model produced by omk_ream_merge.py")
    ap.add_argument("--drop-map", required=True)
    ap.add_argument("--published", default=None,
                    help="optional: published 184e dir; expert tensors must be bit-identical "
                         "when the built arm used --merging none")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--max-experts", type=int, default=0,
                    help="with --published, compare only the first N experts per layer "
                         "(0 = all). A spot-check is enough to catch a systematic change; "
                         "the full sweep reads every shard 3x and is slow.")
    ap.add_argument("--router-prefix", default=None,
                    help="override the router tensor name pattern if the arch differs")
    args = ap.parse_args()

    with open(args.drop_map) as fh:
        drop_map = json.load(fh)

    fmt = (lambda li: args.router_prefix.format(li=li)) if args.router_prefix else router_name

    mismatched, checked = [], 0
    recovered_by_layer = {}
    for li in range(args.layers):
        n = fmt(li)
        base = load_layer_tensors(args.base, [n])[n].float()      # [256, H]
        built = load_layer_tensors(args.built, [n])[n].float()    # [184, H]
        if built.shape[0] >= base.shape[0]:
            raise SystemExit(f"refusing: built L{li} has {built.shape[0]} rows, "
                             f"base has {base.shape[0]} -- not a prune?")

        # exact row matching: for each built row find its base index
        recovered = []
        for r in range(built.shape[0]):
            eq = (base == built[r]).all(dim=1).nonzero().flatten()
            if eq.numel() != 1:
                raise SystemExit(
                    f"refusing: built L{li} row {r} matched {eq.numel()} base rows "
                    f"(expected exactly 1) -- router rows were MODIFIED, so expert identity "
                    f"cannot be recovered this way")
            recovered.append(int(eq[0]))

        expected = sorted(e for e in range(base.shape[0])
                          if e not in set(int(x) for x in drop_map[str(li)]))
        recovered_by_layer[li] = recovered
        checked += 1
        if sorted(recovered) != expected:
            only_built = sorted(set(recovered) - set(expected))
            only_map = sorted(set(expected) - set(recovered))
            mismatched.append((li, only_built[:8], only_map[:8]))
        if recovered != sorted(recovered):
            print(f"NOTE L{li}: surviving experts are not in ascending base order", flush=True)

    print(f"\nlayers checked: {checked}")
    print(f"layers whose recovered keep set != drop map: {len(mismatched)}")
    for li, ob, om in mismatched[:10]:
        print(f"  L{li}: in built not map {ob} | in map not built {om}")

    ok = not mismatched

    if args.published and ok:
        # Position-wise comparison is only meaningful if BOTH checkpoints list survivors in
        # the same order. Both are produced by slicing the same drop map, so both should be
        # ascending by base id; if the built one is not, slot e means different experts in the
        # two files and every "difference" below would be an artefact of the ordering.
        unsorted_layers = [li for li, r in recovered_by_layer.items() if r != sorted(r)]
        if unsorted_layers:
            raise SystemExit(
                f"refusing --published comparison: built survivors are not in ascending base "
                f"order at layers {unsorted_layers[:8]} -- slot-by-slot comparison against the "
                f"published model would compare different experts to each other")

        print("\ncomparing expert tensors against the published 184e ...", flush=True)
        diffs, compared = [], 0
        for li in range(args.layers):
            for w in ("gate_proj", "up_proj", "down_proj"):
                for e in range(min(len(recovered_by_layer[li]), args.max_experts or 10**9)):
                    a = expert_weight(args.built, li, e, w)
                    b = expert_weight(args.published, li, e, w)
                    compared += 1
                    if a.shape != b.shape:
                        diffs.append((f"L{li}.e{e}.{w}",
                                      f"shape {tuple(a.shape)} vs {tuple(b.shape)}"))
                    elif not torch.equal(a, b):
                        md = (a.float() - b.float()).abs().max().item()
                        diffs.append((f"L{li}.e{e}.{w}", f"max|delta|={md:.3e}"))
        print(f"expert tensors compared: {compared}, differing: {len(diffs)}")
        for n, d in diffs[:10]:
            print(f"  {n}: {d}")
        ok = ok and not diffs

    print(f"\n>>> ARM_IDENTITY_{'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
