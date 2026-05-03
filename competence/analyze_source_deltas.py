#!/usr/bin/env python3
"""Per-tensor delta statistics for two source fine-tunes against the base.

Compares Jackrong-v2 (HE-preserving, MBPP-neutral) vs Crow-4B (HE-degrading,
MBPP-improving) against `Qwen/Qwen3.5-4B`. Streams tensors so peak RAM stays
small (one tensor triple at a time, fp32).

Per-tensor signals computed for each source:
  - delta_l2          : ||source - base||_2
  - delta_relative    : ||delta||_2 / ||base||_2
  - delta_mean_abs    : mean(|delta|)
  - sparsity_1pct     : fraction of |delta| > 1% of source's max |delta|
  - sign_flip_rate    : fraction of weights where sign(source) != sign(base)
  - outlier_rate_3s   : fraction of |delta| > 3 * std(delta)

Cross-source signal:
  - delta_cosine      : cos(delta_A, delta_B) — alignment between the two
                        sources' direction of change. >0 = both push same way.
                        <0 = conflict (likely the "merge tax" hot spot).

Layer-type aggregation (attention / mlp / embed / norm / head / other) is
printed at the end so we can see which TYPE of layer carries each signature.

Output: a CSV per-tensor + a printed layer-type summary.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open

LAYER_TYPES = {
    "attn":   re.compile(r"\.(self_attn|linear_attn|attention)\."),
    "mlp":    re.compile(r"\.mlp\."),
    "embed":  re.compile(r"embed_tokens|word_embeddings"),
    "norm":   re.compile(r"(input_layernorm|post_attention_layernorm|final_layernorm|model\.norm|ln_)"),
    "head":   re.compile(r"lm_head"),
    "bias":   re.compile(r"\.bias$"),
}


def classify(name: str) -> str:
    for t, pat in LAYER_TYPES.items():
        if pat.search(name):
            return t
    return "other"


def open_handles(model_dir: Path) -> Dict[str, "safe_open"]:
    """Return {tensor_name: safe_open_handle} for streaming reads."""
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        import json
        wm = json.loads(idx_path.read_text())["weight_map"]
        shards = {s: safe_open(model_dir / s, framework="pt", device="cpu") for s in set(wm.values())}
        return {n: shards[s] for n, s in wm.items()}
    # single-shard: prefer model.safetensors if it exists; else exactly one .safetensors
    candidates = [p for p in model_dir.glob("*.safetensors") if "adapter" not in p.name.lower()]
    main = model_dir / "model.safetensors"
    if main.exists():
        shard = main
    elif len(candidates) == 1:
        shard = candidates[0]
    else:
        raise RuntimeError(f"can't resolve weight map in {model_dir}")
    h = safe_open(shard, framework="pt", device="cpu")
    return {n: h for n in h.keys()}


def get_tensor(handles: Dict[str, "safe_open"], name: str) -> torch.Tensor | None:
    h = handles.get(name)
    if h is None:
        return None
    return h.get_tensor(name)


def per_tensor_stats(base: torch.Tensor, src: torch.Tensor) -> Dict[str, float]:
    base = base.float().flatten()
    src = src.float().flatten()
    delta = src - base
    abs_delta = delta.abs()
    base_norm = base.norm().item() + 1e-12
    delta_norm = delta.norm().item()
    mean_abs = abs_delta.mean().item()
    max_abs = abs_delta.max().item() + 1e-12
    sparsity_1pct = (abs_delta > 0.01 * max_abs).float().mean().item()
    sign_flip = ((torch.sign(src) != torch.sign(base)) & (base != 0)).float().mean().item()
    std_delta = delta.std().item() + 1e-12
    outlier_3s = (abs_delta > 3 * std_delta).float().mean().item()
    return {
        "delta_l2": delta_norm,
        "delta_relative": delta_norm / base_norm,
        "delta_mean_abs": mean_abs,
        "sparsity_1pct": sparsity_1pct,
        "sign_flip_rate": sign_flip,
        "outlier_rate_3s": outlier_3s,
        "_delta_flat": delta,  # passed up so cross-source cosine can use it
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--src-a", required=True, type=Path, help="source A (label A in output)")
    ap.add_argument("--src-b", required=True, type=Path, help="source B")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", required=True, type=Path, help="per-tensor CSV path")
    args = ap.parse_args()

    print(f"[*] base:  {args.base}")
    print(f"[*] {args.label_a}: {args.src_a}")
    print(f"[*] {args.label_b}: {args.src_b}")

    base_h = open_handles(args.base)
    a_h = open_handles(args.src_a)
    b_h = open_handles(args.src_b)

    rows: list[dict] = []
    by_layer_type: Dict[str, list] = {}

    common = sorted(set(base_h) & set(a_h) & set(b_h))
    print(f"[*] {len(common)} tensors in common across all three\n")

    for i, name in enumerate(common):
        base = get_tensor(base_h, name)
        a = get_tensor(a_h, name)
        b = get_tensor(b_h, name)
        if base is None or a is None or b is None:
            continue
        if base.shape != a.shape or base.shape != b.shape:
            continue
        if not base.is_floating_point():
            continue

        sa = per_tensor_stats(base, a)
        sb = per_tensor_stats(base, b)
        # cross-source cosine of the two delta vectors
        da = sa.pop("_delta_flat")
        db = sb.pop("_delta_flat")
        dn = (da.norm() * db.norm()).item() + 1e-12
        cos = (da @ db).item() / dn

        layer = classify(name)
        row = {
            "name": name, "layer_type": layer, "nelem": base.numel(),
            **{f"A_{k}": v for k, v in sa.items()},
            **{f"B_{k}": v for k, v in sb.items()},
            "delta_cosine_AB": cos,
        }
        rows.append(row)
        by_layer_type.setdefault(layer, []).append(row)

        if (i + 1) % 25 == 0 or (i + 1) == len(common):
            print(f"  [{i+1:3}/{len(common)}] {layer:6s} {name[:60]}", flush=True)

    # Write CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        cols = list(rows[0].keys())
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\n[*] wrote per-tensor CSV: {args.out}")

    # Layer-type summary
    print("\n=== Layer-type summary ===")
    print(f"{'layer':<8s} {'n':>5s}   "
          f"{'A_l2':>9s} {'A_relL2':>9s} {'A_flip':>7s} {'A_out3s':>8s}   "
          f"{'B_l2':>9s} {'B_relL2':>9s} {'B_flip':>7s} {'B_out3s':>8s}   "
          f"{'cosAB':>7s}")
    for layer in ("attn", "mlp", "embed", "norm", "head", "bias", "other"):
        rs = by_layer_type.get(layer, [])
        if not rs:
            continue
        n = len(rs)
        def avg(k):
            return sum(r[k] for r in rs) / n
        print(f"{layer:<8s} {n:>5d}   "
              f"{avg('A_delta_l2'):>9.4f} {avg('A_delta_relative'):>9.4f} "
              f"{avg('A_sign_flip_rate'):>7.4f} {avg('A_outlier_rate_3s'):>8.4f}   "
              f"{avg('B_delta_l2'):>9.4f} {avg('B_delta_relative'):>9.4f} "
              f"{avg('B_sign_flip_rate'):>7.4f} {avg('B_outlier_rate_3s'):>8.4f}   "
              f"{avg('delta_cosine_AB'):>7.4f}")

    # Top-conflict tensors (most-negative cosine)
    rows_sorted = sorted(rows, key=lambda r: r["delta_cosine_AB"])
    print("\n=== Top-10 highest-conflict tensors (delta_cosine_AB << 0) ===")
    for r in rows_sorted[:10]:
        print(f"  {r['delta_cosine_AB']:+.3f}  [{r['layer_type']}] {r['name']}")

    print("\n=== Top-10 most-aligned tensors (delta_cosine_AB >> 0) ===")
    for r in rows_sorted[-10:][::-1]:
        print(f"  {r['delta_cosine_AB']:+.3f}  [{r['layer_type']}] {r['name']}")


if __name__ == "__main__":
    sys.exit(main())
