#!/usr/bin/env python3
"""router_pin_downproj_bake.py — GGUF-safe targeted per-pin PES.

Scales ONLY the listed pin experts' ROUTED down_proj slabs by `alpha`, baking a
per-expert mixing-magnitude boost into a tensor that convert_hf_to_gguf actually
exports (`experts.down_proj` -> `ffn_down_exps`). This is the GGUF-safe
equivalent of `router.per_expert_scale[e] *= alpha` for the pins:

  In Gemma4TextRouter the routed contribution of selected expert e is
    routing_weight[e] * per_expert_scale[e] * expert_e(x)
  with expert_e(x) = down_proj_e @ (act(gate_e(x)) * up_e(x)).
  Scaling down_proj_e by alpha makes expert_e(x) -> alpha * expert_e(x), i.e.
  identical to per_expert_scale[e] *= alpha — but per_expert_scale is dropped at
  GGUF conversion (no-op in llama.cpp), whereas down_proj IS exported. So for a
  llama.cpp/GGUF deployment, this down_proj bake is the ONLY way to realise a
  per-expert mixing boost. Used to amplify the 30 agentic-EOG "terminator" pins
  so they reclaim router mixing mass stolen by restored science experts (v8b),
  closing the residual agentic-loop gap without dropping any science capacity.

LAYOUT (verified on the combo header): routed experts are stacked as
  model.language_model.layers.{L}.experts.down_proj  shape [n_experts, hidden, ie]
so slab [idx] is the pruned-expert `idx` down-projection. The pruned index of an
original-128 expert id is its position in the layer's ascending kept list, taken
from the keep-meta ("keep"[str(L)] is sorted ascending == stacked expert order).

Reads the combo's SINGLE model.safetensors directly (no index.json). Reversible
by re-baking with 1/alpha, or simply rebuilding the combo.

Usage:
  router_pin_downproj_bake.py --in-model  <SRC/model.safetensors> \
                              --out-model <PES/model.safetensors> \
                              --keep-meta <keepmeta.json> \
                              --pins "0:2,0:9,..." --alpha 1.3 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path


EXPERTS_DOWN = "model.language_model.layers.{L}.experts.down_proj"


def parse_pins(s: str) -> dict[int, set[int]]:
    pins: dict[int, set[int]] = {}
    for tok in str(s).replace(" ", "").split(","):
        if ":" in tok:
            L, e = tok.split(":")
            pins.setdefault(int(L), set()).add(int(e))
    return pins


def load_keepmap(keep_meta: str) -> dict[int, list[int]]:
    d = json.load(open(keep_meta))
    keep = d["keep"] if "keep" in d else d
    return {int(L): list(v) for L, v in keep.items()}


def read_header(model_path: str) -> dict:
    """Return the safetensors JSON header (tensor -> {shape,dtype,...})."""
    with open(model_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def build_edit_plan(pins, keepmap, header):
    """List of (L, orig_e, pruned_idx, key, n_experts). Asserts validity."""
    plan = []
    for L in sorted(pins):
        key = EXPERTS_DOWN.format(L=L)
        if key not in header:
            sys.exit(f"FATAL: tensor {key} not in header")
        n_experts = header[key]["shape"][0]
        keep = keepmap.get(L)
        if keep is None:
            sys.exit(f"FATAL: layer {L} not in keep-meta")
        if keep != sorted(keep):
            sys.exit(f"FATAL: keep[{L}] is not sorted ascending — index map invalid")
        if len(keep) != n_experts:
            sys.exit(f"FATAL: keep[{L}] has {len(keep)} experts but tensor has {n_experts}")
        for orig_e in sorted(pins[L]):
            if orig_e not in keep:
                sys.exit(f"FATAL: pin {L}:{orig_e} not in kept set (should be force-kept)")
            idx = keep.index(orig_e)
            plan.append((L, orig_e, idx, key, n_experts))
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-model", required=True, help="Source model.safetensors (single file)")
    ap.add_argument("--out-model", help="Destination model.safetensors (required unless --dry-run)")
    ap.add_argument("--keep-meta", required=True, help="keep-meta json with 'keep' map")
    ap.add_argument("--pins", required=True, help="Comma list of original-128 pins 'L:e,...'")
    ap.add_argument("--alpha", type=float, required=True, help="Scale on pin routed down_proj (>0)")
    ap.add_argument("--dry-run", action="store_true", help="Print the edit plan; no load, no write")
    args = ap.parse_args()

    if args.alpha <= 0:
        sys.exit(f"FATAL: --alpha must be > 0 (got {args.alpha})")

    pins = parse_pins(args.pins)
    n_pins = sum(len(v) for v in pins.values())
    keepmap = load_keepmap(args.keep_meta)
    header = read_header(args.in_model)

    plan = build_edit_plan(pins, keepmap, header)
    if len(plan) != n_pins:
        sys.exit(f"FATAL: plan has {len(plan)} edits but {n_pins} pins requested")

    print(f"== pins={n_pins} alpha={args.alpha} layers={sorted(pins)}")
    print(f"{'L':>3} {'orig_e':>6} {'pruned_idx':>10} {'n_exp':>6}")
    for (L, orig_e, idx, key, ne) in plan:
        print(f"{L:>3} {orig_e:>6} {idx:>10} {ne:>6}")
    # group by layer for the rewrite
    by_layer: dict[int, list[int]] = {}
    for (L, _, idx, _, _) in plan:
        by_layer.setdefault(L, []).append(idx)
    print(f"== distinct layers edited: {len(by_layer)}; total slabs scaled: {len(plan)}")

    if args.dry_run:
        print("== dry-run: no tensors loaded, no file written.")
        return 0

    if not args.out_model:
        sys.exit("FATAL: --out-model required when not --dry-run")

    # heavy imports only for the real run
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, "torch.Tensor"] = {}
    meta: dict = {}
    targets = {EXPERTS_DOWN.format(L=L) for L in by_layer}
    print(f"== loading {len(header)-(1 if '__metadata__' in header else 0)} tensors from {args.in_model}")
    with safe_open(args.in_model, framework="pt") as f:
        m = f.metadata()
        if m:
            meta = dict(m)
        for k in f.keys():
            t = f.get_tensor(k)
            if k in targets:
                t = t.clone()  # ensure writable contiguous copy
            tensors[k] = t

    scaled = 0
    for L, idxs in by_layer.items():
        key = EXPERTS_DOWN.format(L=L)
        t = tensors[key]
        dt = t.dtype
        for idx in idxs:
            t[idx] = (t[idx].to(torch.float32) * args.alpha).to(dt)
            scaled += 1
        tensors[key] = t
    if scaled != n_pins:
        sys.exit(f"FATAL: scaled {scaled} slabs but expected {n_pins}")
    print(f"== scaled {scaled} pin slabs by alpha={args.alpha}")

    print(f"== writing {out_path}")
    save_file(tensors, str(out_path), metadata=meta or None)
    sz = out_path.stat().st_size
    print(f"== done: {out_path} ({sz} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
