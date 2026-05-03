#!/usr/bin/env python3
"""Build M6 hybrid importance scores: Fisher on attention, LRP on MLP.

Routing rules (Qwen3.5 / Qwen3 family layer naming):
- Attention tensors → Fisher
    - .self_attn.q_proj, .k_proj, .v_proj, .o_proj
    - .linear_attn.* (Qwen3.5 hybrid linear-attn variants, if present)
- MLP tensors → LRP
    - .mlp.gate_proj, .mlp.up_proj, .mlp.down_proj
- Embeddings / norms / lm_head → take whichever is present (Fisher preferred,
  fall back to LRP). These are typically untouched or near-base anyway.

Output is a single safetensors per source that the mergekit ex-LRP method
can consume directly via `lrp_scores: <path>`. The merger only sees one
importance signal per tensor — it doesn't know it's hybrid.

Both Fisher and LRP must contain the same tensor key set (or close) — script
emits a diff report and skips tensors missing in the chosen signal (the merger
will treat them as base passthrough).

Each tensor is rank-normalised to [0, 1] before being written, so the
density-thresholding inside mergekit applies the same comparison budget to both
signal types regardless of their raw magnitude scales.

Usage:
  python build_hybrid_fisher_lrp.py FISHER.safetensors LRP.safetensors OUT.safetensors
"""
import sys
import re

import torch
from safetensors.torch import load_file, save_file

ATTN_PATTERN = re.compile(r"\.(self_attn|linear_attn)\.")
MLP_PATTERN = re.compile(r"\.mlp\.")


def rank_normalize(t: torch.Tensor) -> torch.Tensor:
    """Map values to [0, 1] by rank — preserves order, removes scale."""
    flat = t.flatten().float()
    ranks = torch.argsort(torch.argsort(flat))
    out = ranks.float() / max(len(flat) - 1, 1)
    return out.reshape(t.shape)


def _normalize_to_lrp_namespace(d: dict) -> dict:
    """Promote Fisher's `model.<X>` keys into LRP's `model.language_model.<X>`
    namespace. Qwen3.5 hybrid bases ship the language_model under that prefix
    (LRP capture from PR #682's hooks reflects the actual module path), but
    Fisher capture (precompute_fisher.py) uses HF's flattened state-dict view.
    Without this, Fisher and LRP have zero key overlap and the M6 hybrid
    degenerates into LRP-only / base-passthrough.
    """
    out = {}
    for k, v in d.items():
        if k.startswith("model.") and not k.startswith("model.language_model."):
            new = "model.language_model." + k[len("model."):]
        else:
            new = k
        out[new] = v
    return out


def main(fisher_path: str, lrp_path: str, out_path: str) -> None:
    print(f"[*] loading Fisher: {fisher_path}")
    fisher = load_file(fisher_path)
    print(f"    {len(fisher)} tensors (raw)")
    fisher = _normalize_to_lrp_namespace(fisher)
    print(f"    {len(fisher)} tensors (after namespace promotion to model.language_model.*)")
    print(f"[*] loading LRP:    {lrp_path}")
    lrp = load_file(lrp_path)
    print(f"    {len(lrp)} tensors")

    f_keys = set(fisher.keys())
    l_keys = set(lrp.keys())
    common = f_keys & l_keys
    f_only = f_keys - l_keys
    l_only = l_keys - f_keys
    print(f"[*] common: {len(common)}  fisher-only: {len(f_only)}  lrp-only: {len(l_only)}")

    out: dict[str, torch.Tensor] = {}
    counts = {"attn_fisher": 0, "mlp_lrp": 0, "other_fisher": 0, "other_lrp": 0, "skipped": 0}
    sample = {}

    all_keys = f_keys | l_keys
    for k in sorted(all_keys):
        is_attn = bool(ATTN_PATTERN.search(k))
        is_mlp = bool(MLP_PATTERN.search(k))

        if is_attn and k in fisher:
            t = rank_normalize(fisher[k])
            counts["attn_fisher"] += 1
            sample.setdefault("attn", k)
        elif is_mlp and k in lrp:
            t = rank_normalize(lrp[k])
            counts["mlp_lrp"] += 1
            sample.setdefault("mlp", k)
        elif k in fisher:
            t = rank_normalize(fisher[k])
            counts["other_fisher"] += 1
            sample.setdefault("other", k)
        elif k in lrp:
            t = rank_normalize(lrp[k])
            counts["other_lrp"] += 1
            sample.setdefault("other", k)
        else:
            counts["skipped"] += 1
            continue

        out[k] = t.contiguous().to(torch.float32)

    print(f"[*] routing breakdown: {counts}")
    print(f"[*] sample keys: {sample}")
    print(f"[*] writing {out_path} ({len(out)} tensors)")
    save_file(out, out_path)
    print("[done]")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    main(*sys.argv[1:4])
