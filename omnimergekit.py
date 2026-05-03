#!/usr/bin/env python3
"""
Minimal DARE-TIES merger for Qwen3.5-27B (or any architecture that mergekit
doesn't understand). Processes tensors one at a time, matching by name across
the base + source models. Safe for hybrid architectures (linear_attn layers,
vision towers, multi-modal projectors) — anything present in both base and
sources gets merged.

DARE-TIES (density=d, K sources):
  1) For each source i:  delta_i = source_i - base
  2) DARE drop: mask_i = Bernoulli(d), delta_i *= mask_i / d  (rescale)
  3) TIES sign consensus: keep only delta_i entries whose sign matches the
     sign of sum_i (mask_i * delta_i), zero out the rest
  4) TIES merge: merged_delta = sum_i (kept_delta_i) / count_nonzero (per elem)
     (equivalent to weighted average of surviving deltas)
  5) merged = base + merged_delta

Notes:
- Equal weights are used across sources by default.
- Tensors present in base but not in all sources get copied from base.
- Tensors present in sources but not in base are skipped (can't compute delta).
- int-typed tensors are copied from base (indices, int buffers, etc).

Templates:
  # Omnimerge v1 (DARE-TIES baseline)
  python dare_ties_merge.py \\
      --base /path/to/Qwen3.5-27B \\
      --source claude-distill --source esper --source gemini \\
      --output /path/to/omnimerge-v1 \\
      --method dare_ties --density 0.53 --weights 0.40,0.35,0.25 --seed 42

  # Omnimerge v2 (OBIM masking + DAREx rescaling + EMR election)
  python dare_ties_merge.py \\
      --base /path/to/Qwen3.5-27B \\
      --source claude-distill --source esper --source gemini \\
      --output /path/to/omnimerge-v2 \\
      --method omnimerge_v2 --density 0.53 --weights 0.40,0.35,0.25 \\
      --darex-q 0.75 --seed 42

  # Qwen3.6-27B MLP-skip merge (mandatory for Qwen3.6 — its <think>-emission
  # policy lives in MLP attractors and is flipped by even small MLP deltas;
  # see feedback_qwen3_6_merge_policy_fragility.md):
  python dare_ties_merge.py \\
      --base /path/to/Qwen3.6-27B \\
      --source rico03 --source esper3.1 --source kai-os-anchor \\
      --output /path/to/omnimerge-v5-mlp-skip \\
      --method omnimerge_v2 --density 0.53 --weights 0.40,0.35,0.25 \\
      --darex-q 0.75 --seed 42 \\
      --skip-patterns 'mlp.gate_proj,mlp.up_proj,mlp.down_proj'
"""
import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_weight_map(model_dir: Path) -> Tuple[Dict[str, str], List[str]]:
    """Return (tensor_name -> relative_shard_filename, ordered_tensor_list)."""
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        with open(idx_path) as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]
        # preserve original ordering for stability
        names = list(weight_map.keys())
        return weight_map, names

    # Fallback: single-shard model (no index). Exclude PEFT adapter files
    # (e.g. adapter_model.safetensors) — those are LoRA adapters, not the
    # main weights. crow-4b ships both `model.safetensors` and
    # `adapter_model.safetensors`; we want the former.
    shards = sorted(p for p in model_dir.glob("*.safetensors")
                    if "adapter" not in p.name.lower())
    main = model_dir / "model.safetensors"
    if main.exists():
        shard = main
    elif len(shards) == 1:
        shard = shards[0]
    else:
        raise RuntimeError(
            f"{model_dir}: no index.json and {len(shards)} shards — can't resolve weight map"
        )
    with safe_open(shard, framework="pt", device="cpu") as f:
        names = list(f.keys())
    weight_map = {name: shard.name for name in names}
    return weight_map, names


def open_shard_handles(model_dir: Path, weight_map: Dict[str, str]) -> Dict[str, "safe_open"]:
    """Open one safe_open handle per shard file. Caller must close them (they don't
    support context-manager-less closing — we just rely on GC / process end)."""
    handles = {}
    for shard_name in set(weight_map.values()):
        handles[shard_name] = safe_open(model_dir / shard_name, framework="pt", device="cpu")
    return handles


def get_tensor(handles: Dict[str, "safe_open"], weight_map: Dict[str, str], name: str) -> Optional[torch.Tensor]:
    shard_name = weight_map.get(name)
    if shard_name is None:
        return None
    return handles[shard_name].get_tensor(name)


CHUNK_ELEMENTS = 50_000_000  # 50M elements ≈ 200 MB in fp32 per view


# Turbo enhancement (ported from PR #682 turbo head): tensors that are
# semantically critical to the network (norms, embeddings, lm_head, biases)
# should NOT be sparsified — the merge math is fundamentally density-based,
# but for these layers any density < 1 destroys the per-token / per-vocab
# bias structure and degrades generation immediately. Force density=1.0
# regardless of the user-supplied --density when name matches.
_CRITICAL_LAYER_TOKENS = ("norm", "embed", "ln_", "lm_head", "head", "bias")


def _is_critical_layer(name: str) -> bool:
    n = name.lower()
    return any(tok in n for tok in _CRITICAL_LAYER_TOKENS)


# ────────────────────────────────────────────────────────────────────────────
# M7 mergetime detector — independently gated by --m7-detector.
#
# Empirical motivation (delta_analysis_jackrong_vs_crow.csv, this project):
#   - Sources with relL2 ~0.001 (Jackrong-style soft distill) preserve base
#     capability; sources with relL2 ~0.03 (Crow-style aggressive distill) hurt
#     HumanEval. Sign-flip rate scales with relL2 ~35×.
#   - Cross-source delta cosine is essentially 0 everywhere (sources move in
#     independent directions). Top-aligned tensors (cos > +0.003) are MLP
#     gate_proj/up_proj middle layers — where merge GAINS concentrate.
#
# M7 detector: per-tensor pre-merge adjustment:
#   1. Norm-clamp:    if relL2_i > τ_norm,  scale delta_i down to τ_norm·||base||.
#   2. Sign-flip gate: if flip_rate_i > τ_flip, multiply weight_i by exp(-100·flip_rate_i).
#   3. Consensus bonus: if cos(Δ_A, Δ_B) > τ_cos, multiply both weights by 1.2.
# Conservative defaults from the data analysis above.
# ────────────────────────────────────────────────────────────────────────────
_M7_TAU_NORM = 0.02      # per-source ||delta||/||base|| cap (attn / default)
_M7_TAU_NORM_MLP = 0.10  # M7v2: looser cap on MLP — Crow's MBPP-helping deltas concentrate here
_M7_TAU_FLIP = 0.01      # 1% sign-flip rate triggers per-tensor downweight
_M7_TAU_COS  = 0.003     # consensus cosine threshold for cross-source bonus
_M7_BONUS    = 1.2

# M7v2: layer-type pattern set for relaxed treatment
_M7_MLP_PAT = re.compile(r"\.mlp\.")


def _m7_adjust(
    base: torch.Tensor,
    sources: List[torch.Tensor],
    weights: List[float],
    tensor_name: Optional[str] = None,
    layer_aware: bool = False,
    tau_norm_override: Optional[float] = None,
    flip_strength: float = 100.0,
) -> Tuple[List[torch.Tensor], List[float], dict]:
    """Apply norm-clamp + sign-flip gate + consensus bonus pre-merge.

    Returns (adjusted_sources, adjusted_weights, stats_for_logging).
    All tensors stay on `base.device`; full-tensor scan is one pass per source.
    """
    base_f = base.float().flatten()
    base_norm = base_f.norm().item() + 1e-12
    base_sign = torch.sign(base_f)

    deltas = []
    rel_l2 = []
    flip = []
    for s in sources:
        s_f = s.float().flatten()
        d = s_f - base_f
        deltas.append(d)
        rel_l2.append(d.norm().item() / base_norm)
        flip.append(((torch.sign(s_f) != base_sign) & (base_f != 0)).float().mean().item())

    # M7v2: looser threshold on MLP tensors when layer_aware=True
    # M7v3: explicit τ_norm override (CLI), uniform across layers
    is_mlp = bool(tensor_name and _M7_MLP_PAT.search(tensor_name))
    if tau_norm_override is not None:
        tau_norm = tau_norm_override
    elif layer_aware and is_mlp:
        tau_norm = _M7_TAU_NORM_MLP
    else:
        tau_norm = _M7_TAU_NORM

    # 1. Norm-clamp on delta (rescale source toward base where excessive)
    new_sources = []
    clamped_count = 0
    for i, (s, d, r) in enumerate(zip(sources, deltas, rel_l2)):
        if r > tau_norm:
            scale = tau_norm / r
            new_d = d * scale
            new_sources.append((base.float().flatten() + new_d).reshape(base.shape).to(s.dtype))
            clamped_count += 1
        else:
            new_sources.append(s)

    # 2. Sign-flip gate — adjust weights. M7v2: skip on MLP layers (Crow's
    # MLP sign-flips carry MBPP gain — don't punish them).
    adj_w = list(weights)
    flip_gated = 0
    if not (layer_aware and is_mlp):
        for i, fr in enumerate(flip):
            if fr > _M7_TAU_FLIP:
                adj_w[i] *= float(torch.exp(torch.tensor(-flip_strength * fr)).item())
                flip_gated += 1

    # 3. Consensus bonus — only meaningful for K=2 (K>2 = pairwise, skip)
    cos_ab = None
    bonus_applied = False
    if len(deltas) == 2:
        a, b = deltas[0], deltas[1]
        denom = (a.norm() * b.norm()).item() + 1e-12
        cos_ab = (a @ b).item() / denom
        if cos_ab > _M7_TAU_COS:
            adj_w[0] *= _M7_BONUS
            adj_w[1] *= _M7_BONUS
            bonus_applied = True

    stats = {
        "rel_l2": rel_l2,
        "flip": flip,
        "cos_ab": cos_ab,
        "clamped": clamped_count,
        "flip_gated": flip_gated,
        "bonus": bonus_applied,
    }
    # free big intermediates
    del deltas, base_f, base_sign
    return new_sources, adj_w, stats


def _dare_drop(delta: torch.Tensor, density: float, generator: torch.Generator) -> torch.Tensor:
    """DARE: uniform random Bernoulli drop, rescale survivors by 1/density.

    Preserves expectation: E[masked] == delta. High variance at low density.
    """
    if delta.device.type == "cuda":
        mask = (torch.empty(delta.shape, device="cpu").uniform_(generator=generator) < density).to(delta.device)
    else:
        mask = torch.empty_like(delta).uniform_(generator=generator) < density
    return torch.where(mask, delta / density, torch.zeros_like(delta))


def _magprune_drop(delta: torch.Tensor, density: float, epsilon: float,
                   generator: torch.Generator) -> torch.Tensor:
    """DELLA MAGPRUNE (arXiv 2406.11617 §3): magnitude-ranked drop.

    Instead of dropping every entry with the same probability p=1-density, rank
    delta entries by |value| within this chunk. Smallest-magnitude entries get
    drop prob `p + ε/2` (more likely to be dropped), largest get `p - ε/2`
    (more likely to be kept). Keep probability per element is `(1 - p_i)` =
    `density_i`, and survivors are rescaled by `1 / density_i` elementwise.

    This preserves the expectation of each delta individually while biasing
    sparsification toward the informative (high-magnitude) entries. Per DELLA
    paper and Qwen-Coder merging study, beats uniform DARE by +1-3 pp on
    math/code merges of same-base specialists.
    """
    n = delta.nelement()
    p_base = 1.0 - density  # baseline drop prob
    # Rank by |value|: r_i ∈ {0..n-1}, 0 = smallest |delta|
    # torch.argsort gives permutation indices; we want inverse (ranks)
    abs_flat = delta.abs().flatten()
    sort_idx = torch.argsort(abs_flat, stable=True)
    ranks = torch.empty_like(sort_idx)
    ranks[sort_idx] = torch.arange(n, device=delta.device)
    # Per-element drop prob: smallest |delta| gets p_base + ε/2, largest gets p_base - ε/2
    # Linear interpolation across ranks
    delta_frac = (ranks.to(torch.float32) / max(1, n - 1)) - 0.5  # [-0.5, +0.5]
    p_i = p_base + epsilon * (-delta_frac) * 2.0  # smallest rank → p_base + ε, largest → p_base - ε
    # Wait, invert: rank 0 (smallest |delta|) should have HIGHER drop prob
    # delta_frac at rank 0 = -0.5 → -delta_frac * 2 = +1 → p_i = p_base + ε ✓
    # delta_frac at rank n-1 = +0.5 → -delta_frac * 2 = -1 → p_i = p_base - ε ✓
    # Clamp into [0, 1) to be safe
    p_i = p_i.clamp(min=0.0, max=0.9999).view_as(delta)
    density_i = 1.0 - p_i  # per-element KEEP probability

    # Bernoulli mask with per-element prob (1 - p_i). CPU-generator workaround for CUDA tensors.
    if delta.device.type == "cuda":
        rand = torch.empty(delta.shape, device="cpu").uniform_(generator=generator).to(delta.device)
    else:
        rand = torch.empty_like(delta).uniform_(generator=generator)
    mask = rand < density_i
    # Rescale survivors by 1/density_i elementwise (preserves expectation)
    out = torch.where(mask, delta / density_i.clamp(min=1e-4), torch.zeros_like(delta))
    return out


def _apply_drop(delta: torch.Tensor, method: str, density: float,
                epsilon: float, generator: torch.Generator) -> torch.Tensor:
    """Dispatch to the appropriate drop method for the current merge method."""
    if method == "task_arithmetic":
        # No drop, no rescale — keep everything
        return delta
    if method == "della":
        return _magprune_drop(delta, density, epsilon, generator)
    # dare_ties, dare_linear → uniform DARE
    return _dare_drop(delta, density, generator)


def _omnimerge_v2_chunk(
    base_chunk: torch.Tensor,  # fp32, flat 1D
    source_chunks: List[torch.Tensor],  # fp32, flat 1D
    weights: List[float],
    density: float,
    darex_q: float,
    fisher_chunks: Optional[List[torch.Tensor]] = None,
    features: Optional[set] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Omnimerge v2: configurable enhancements over DARE-TIES.

    Features (toggled via `features` set):
      obim:   OBIM-lite magnitude top-k masking (replaces random DARE drop)
      darex:  DAREx rescaling (q instead of density)
      emr:    EMR election (max abs instead of TIES mean)
      fisher: Fisher per-parameter weighting (requires fisher_chunks)

    When a feature is disabled, the corresponding step falls back to DARE-TIES
    behavior. This enables ablation studies to test each enhancement individually.
    """
    if features is None:
        features = {"obim", "darex", "emr"}

    N = base_chunk.nelement()
    k = max(1, int(density * N))

    # Fisher weights (if enabled and available)
    fisher_weights = None
    if "fisher" in features and fisher_chunks is not None:
        # Turbo enhancement: shape-mismatch fallback. Some Fisher/LRP signal
        # files have stale shapes (e.g. from an earlier model revision); silently
        # fall back to uniform weighting rather than crashing the whole merge.
        bad = any(f.shape != base_chunk.shape for f in fisher_chunks)
        if bad:
            fisher_chunks = None
        else:
            fisher_stack = torch.stack(fisher_chunks, dim=0)  # [K, N]
            fisher_weights = fisher_stack / fisher_stack.sum(dim=0, keepdim=True).clamp(min=1e-8)
            del fisher_stack

    # Step 1+2: Compute deltas with masking + rescaling
    masked = []
    for i, s in enumerate(source_chunks):
        delta = s - base_chunk
        if density < 1.0:
            if "obim" in features:
                # OBIM-lite: deterministic top-k by |delta|
                _, top_idx = torch.topk(delta.abs(), k, largest=True, sorted=False)
                mask = torch.zeros(N, dtype=torch.bool, device=delta.device)
                mask[top_idx] = True
                del top_idx
            else:
                # Fallback: random DARE Bernoulli drop
                # generator is CPU-only; for CUDA tensors, generate on CPU and move
                mask = (torch.empty(delta.shape, device="cpu").uniform_(generator=generator) < density).to(delta.device)

            if "darex" in features:
                # DAREx: rescale by 1/q
                delta = torch.where(mask, delta / darex_q, torch.zeros_like(delta))
            else:
                # Fallback: DARE rescale by 1/density
                delta = torch.where(mask, delta / density, torch.zeros_like(delta))
            del mask

        # Apply weights
        if fisher_weights is not None:
            masked.append(delta * fisher_weights[i])
        else:
            masked.append(delta * weights[i])
        del delta

    if fisher_weights is not None:
        del fisher_weights

    stack = torch.stack(masked, dim=0)  # [K, N]
    del masked

    if "emr" in features:
        # EMR election: sign from weighted sum, amplitude from max abs
        weighted_sum = stack.sum(dim=0)
        consensus_sign = torch.sign(weighted_sum)
        del weighted_sum

        max_abs, _ = stack.abs().max(dim=0)
        merged_delta = consensus_sign * max_abs
        del max_abs

        # Noise filter
        any_agree = (torch.sign(stack) == consensus_sign.unsqueeze(0)).any(dim=0)
        merged_delta = torch.where(any_agree, merged_delta, torch.zeros_like(merged_delta))
        del consensus_sign, any_agree
    else:
        # Fallback: TIES sign consensus + weighted mean
        w_tensor = torch.tensor(weights, dtype=torch.float32, device=stack.device).view(-1, 1)
        summed = stack.sum(dim=0)
        consensus_sign = torch.sign(summed)
        del summed
        sign_stack = torch.sign(stack)
        keep = sign_stack == consensus_sign.unsqueeze(0)
        del sign_stack, consensus_sign
        kept = torch.where(keep, stack, torch.zeros_like(stack))
        w_mask = keep.to(torch.float32) * w_tensor
        del keep
        sum_w = w_mask.sum(dim=0).clamp(min=1e-8)
        del w_mask
        merged_delta = kept.sum(dim=0) / sum_w
        del kept, sum_w

    del stack
    out = base_chunk + merged_delta
    del merged_delta
    return out


def _merge_chunk(
    base_chunk: torch.Tensor,  # fp32
    source_chunks: List[torch.Tensor],  # fp32
    weights: List[float],  # per-source weights, same order as source_chunks
    method: str,  # dare_ties | dare_linear | task_arithmetic | della | omnimerge_v2
    density: float,
    epsilon: float,
    generator: torch.Generator,
    darex_q: Optional[float] = None,
    fisher_chunks: Optional[List[torch.Tensor]] = None,
    v2_features: Optional[set] = None,
) -> torch.Tensor:
    """Merge a single flat 1D chunk with the requested method.

    Methods:
      - dare_ties: DARE drop + TIES sign consensus (the original, our v1 baseline)
      - dare_linear: DARE drop + weighted linear merge, NO sign consensus
      - task_arithmetic: no drop, no sign consensus, just base + Σ(w_i · Δ_i)
      - della: MAGPRUNE drop (magnitude-ranked) + TIES sign consensus
      - omnimerge_v2: magnitude top-k (OBIM-lite) + DAREx rescaling + EMR election
    """
    if method == "omnimerge_v2":
        return _omnimerge_v2_chunk(base_chunk, source_chunks, weights,
                                   density, darex_q or density,
                                   fisher_chunks=fisher_chunks,
                                   features=v2_features,
                                   generator=generator)

    # Compute and drop deltas per source
    masked = []
    for s in source_chunks:
        delta = s - base_chunk  # fp32, ~200 MB per chunk
        delta = _apply_drop(delta, method, density, epsilon, generator)
        masked.append(delta)
        del delta

    # Apply source weights in-place
    for i in range(len(masked)):
        masked[i] = masked[i] * weights[i]

    # Stack for elementwise ops
    stack = torch.stack(masked, dim=0)  # [K, N]
    del masked

    if method in ("task_arithmetic", "dare_linear"):
        # Pure weighted linear merge: base + Σ(w_i · Δ_i)
        # Note: weights should sum to 1.0 for unbiased output scale
        merged_delta = stack.sum(dim=0)
        del stack
        out = base_chunk + merged_delta
        del merged_delta
        return out

    # dare_ties / della: TIES sign consensus on the weighted sum
    w_tensor = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
    summed = stack.sum(dim=0)
    consensus_sign = torch.sign(summed)
    del summed
    sign_stack = torch.sign(stack)
    keep = sign_stack == consensus_sign.unsqueeze(0)
    del sign_stack, consensus_sign

    kept = torch.where(keep, stack, torch.zeros_like(stack))
    del stack

    # Weighted average: sum(kept_i) / sum(w_i · alive_i)
    # kept already has w_i baked in, so numerator = kept.sum(dim=0)
    w_mask = keep.to(torch.float32) * w_tensor  # [K, N]
    del keep
    sum_weights = w_mask.sum(dim=0).clamp(min=1e-8)
    del w_mask

    merged_delta = kept.sum(dim=0) / sum_weights
    del kept, sum_weights

    out = base_chunk + merged_delta
    del merged_delta
    return out


# Backwards-compat alias: old callers used _dare_ties_chunk
def _dare_ties_chunk(base_chunk, source_chunks, weights, density, generator):
    return _merge_chunk(base_chunk, source_chunks, weights, "dare_ties", density, 0.0, generator)


def dare_ties_merge_tensor(
    base: torch.Tensor,
    sources: List[torch.Tensor],
    density: float,
    generator: torch.Generator,
    weights: Optional[List[float]] = None,
    method: str = "dare_ties",
    epsilon: float = 0.1,
    darex_q: Optional[float] = None,
    device: str = "cpu",
    fisher_tensors: Optional[List[torch.Tensor]] = None,
    v2_features: Optional[set] = None,
    tensor_name: Optional[str] = None,
    pr682_turbo: bool = False,
    m7_detector: bool = False,
    m7_layer_aware: bool = False,
    m7_tau_norm: Optional[float] = None,
    m7_flip_strength: float = 100.0,
) -> torch.Tensor:
    """Apply DARE-TIES to a single tensor (returns merged result, same shape/dtype).

    Chunks the computation along the flat view so peak RAM per tensor is bounded
    by CHUNK_ELEMENTS regardless of the tensor's total size. Critical for large
    tensors like lm_head/embed_tokens (1.27B elements on Qwen3.5-27B = 4.7 GB fp32
    per tensor, which would need ~100+ GB working set un-chunked).

    When device="cuda", chunks are moved to GPU for computation (topk, stack, EMR
    election) then moved back. ~5-10x faster than CPU for omnimerge_v2.
    """
    if weights is None:
        weights = [1.0 / len(sources)] * len(sources)
    assert len(weights) == len(sources), f"weights ({len(weights)}) must match sources ({len(sources)})"

    orig_dtype = base.dtype
    shape = base.shape
    n = base.nelement()

    # Turbo enhancement (gated by --pr682-turbo): norms / embeddings /
    # lm_head / biases tolerate density<1 poorly; force density=1 (no
    # sparsification, just weighted delta merge) regardless of caller-supplied
    # --density. Applies to all merge methods. Default off so existing recipes
    # (M1/M2/M3/M5) reproduce byte-identically.
    if (pr682_turbo and tensor_name is not None
            and _is_critical_layer(tensor_name)):
        density = 1.0
        if darex_q is not None and darex_q < 1.0:
            darex_q = 1.0

    # M7 mergetime detector (gated by --m7-detector): per-tensor pre-merge
    # adjustment using global delta statistics. Skip critical layers — they
    # are protected by pr682_turbo or untouched anyway. Operates on the full
    # tensor (norm/cos are global, not chunkable).
    if m7_detector and not (tensor_name is not None and _is_critical_layer(tensor_name)):
        sources, weights, m7_stats = _m7_adjust(
            base, sources, weights,
            tensor_name=tensor_name, layer_aware=m7_layer_aware,
            tau_norm_override=m7_tau_norm, flip_strength=m7_flip_strength,
        )
        if os.environ.get("MERGE_DEBUG", "") == "1" and (m7_stats["clamped"] or m7_stats["flip_gated"] or m7_stats["bonus"]):
            print(f"     [m7] {tensor_name}: clamp={m7_stats['clamped']} "
                  f"flip_gate={m7_stats['flip_gated']} bonus={m7_stats['bonus']} "
                  f"cos={m7_stats['cos_ab']}", flush=True)

    # Output buffer in original dtype, flat
    out_flat = torch.empty(n, dtype=orig_dtype)

    base_flat = base.reshape(-1)
    source_flats = [s.reshape(-1) for s in sources]
    fisher_flats = [f.reshape(-1) for f in fisher_tensors] if fisher_tensors else None

    for start in range(0, n, CHUNK_ELEMENTS):
        end = min(start + CHUNK_ELEMENTS, n)
        base_chunk = base_flat[start:end].to(dtype=torch.float32, device=device).contiguous()
        source_chunks = [s[start:end].to(dtype=torch.float32, device=device).contiguous() for s in source_flats]
        f_chunks = [f[start:end].to(dtype=torch.float32, device=device).contiguous() for f in fisher_flats] if fisher_flats else None
        merged_chunk = _merge_chunk(base_chunk, source_chunks, weights, method, density, epsilon, generator, darex_q=darex_q, fisher_chunks=f_chunks, v2_features=v2_features)
        out_flat[start:end] = merged_chunk.to(dtype=orig_dtype, device="cpu")
        del base_chunk, source_chunks, merged_chunk
        if device != "cpu":
            torch.cuda.empty_cache()
        gc.collect()

    return out_flat.reshape(shape)


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path, help="Base model directory (HF safetensors)")
    ap.add_argument("--source", required=True, action="append", type=Path,
                    help="Source fine-tune directory. Pass multiple times for multi-source.")
    ap.add_argument("--output", required=True, type=Path, help="Output merged model directory")
    ap.add_argument("--method", type=str, default="dare_ties",
                    choices=["dare_ties", "dare_linear", "task_arithmetic", "della", "omnimerge_v2"],
                    help="Merge method. dare_ties=DARE drop + TIES sign consensus (original). "
                         "dare_linear=DARE drop + weighted linear (no sign consensus). "
                         "task_arithmetic=no drop, no sign, just base + Σ(wΔ). "
                         "della=MAGPRUNE drop (magnitude-ranked) + TIES sign consensus. "
                         "omnimerge_v2=magnitude top-k (OBIM-lite) + DAREx rescaling + EMR election.")
    ap.add_argument("--density", type=float, default=0.53,
                    help="Drop density (keep fraction). Ignored for task_arithmetic. Default 0.53 for dare_ties, recommend 0.7 for della.")
    ap.add_argument("--epsilon", type=float, default=0.1,
                    help="DELLA MAGPRUNE ε (magnitude-rank drop spread). Default 0.1. Only used when --method della.")
    ap.add_argument("--darex-q", type=float, default=None,
                    help="DAREx rescaling factor q (omnimerge_v2 only). Survivors are divided by q "
                         "instead of density. q > density means less rescaling (lower variance). "
                         "Default: density (equivalent to DARE). Try 0.7-0.85 for DAREx benefit.")
    ap.add_argument("--fisher", type=str, default=None,
                    help="Comma-separated paths to Fisher score safetensors files, one per source "
                         "(omnimerge_v2 only). Pre-computed with precompute_fisher.py. Enables "
                         "per-parameter adaptive weighting instead of fixed source weights.")
    ap.add_argument("--v2-features", type=str, default="obim,darex,emr",
                    help="Comma-separated list of omnimerge_v2 features to enable. "
                         "Options: obim (magnitude top-k masking), darex (DAREx rescaling), "
                         "emr (EMR election), fisher (requires --fisher). "
                         "Default: obim,darex,emr. Use for ablation studies.")
    ap.add_argument("--weights", type=str, default=None,
                    help="Comma-separated per-source weights (e.g. '0.4,0.25,0.25,0.1'). "
                         "Must match number of --source flags. Default: equal weights.")
    ap.add_argument("--no-auto-mlp-skip", action="store_true",
                    help="Disable automatic MLP-skip on Qwen3.6+ bases. By default, this script reads "
                         "<base>/config.json and auto-adds 'mlp.gate_proj,mlp.up_proj,mlp.down_proj' to "
                         "--skip-patterns when output_gate_type is non-null (Qwen3.6 family). Pass this "
                         "flag to override and merge MLPs anyway (e.g. for ablation studies).")
    ap.add_argument("--skip-patterns", type=str, default=None,
                    help="Comma-separated substring patterns. Any tensor whose full name contains "
                         "ANY of these substrings is COPIED FROM BASE instead of merged. Use this when "
                         "merging onto a base whose policy is fragile to certain weight perturbations. "
                         "For Qwen3.6-27B same-base merges, pass "
                         "--skip-patterns 'mlp.gate_proj,mlp.up_proj,mlp.down_proj' — Qwen3.6's "
                         "<think>-emission policy lives in MLP attractor structure and is flipped by "
                         "even 1-2%% rel-L2 perturbations there. Confirmed by v4-MLP-passthrough "
                         "isolation test (0%% leak, mbpp pass@1 50%% vs full-merge 20%%). See "
                         "memory/feedback_qwen3_6_merge_policy_fragility.md.")
    ap.add_argument("--pr682-turbo", action="store_true",
                    help="TURBO PROTECTION (ported from mergekit PR #682 turbo head): "
                         "force density=1.0 for tensors matching norm/embed/ln_/lm_head/"
                         "head/bias regardless of --density. These layers' per-token / "
                         "per-vocab structure tolerates sparsification poorly and degrades "
                         "generation when merged at density<1. Default OFF for "
                         "byte-identical reproduction of M1/M2/M3/M5 recipes; turn on for "
                         "new recipes that aim to preserve base-model behavior on critical "
                         "layers (recommended).")
    ap.add_argument("--m7-tau-norm", type=float, default=None,
                    help="M7v3: explicit τ_norm for the M7 detector clamp (default 0.02). "
                         "Setting this overrides --m7-layer-aware. Try 0.05 for a "
                         "mid-strength clamp between M7 (0.02, HE-favoring) and M7v2-MLP (0.10).")
    ap.add_argument("--m7-flip-strength", type=float, default=100.0,
                    help="M7v3: sign-flip-gate exponent multiplier (default 100). "
                         "Lower = gentler downweight per flip. Try 50 for half-strength.")
    ap.add_argument("--m7-layer-aware", action="store_true",
                    help="M7v2: layer-type-aware clamping. Loosen norm-clamp on "
                         "MLP tensors (τ=0.10 vs τ=0.02 for attn) and skip the "
                         "sign-flip gate on MLP. Goal: keep HE gain (attn-side "
                         "Crow clamping) while preserving MBPP gain (MLP-side "
                         "Crow signal). Has no effect unless --m7-detector is also passed.")
    ap.add_argument("--m7-detector", action="store_true",
                    help="M7 MERGETIME DETECTOR (NOT part of PR #682; project-internal). "
                         "Per-tensor pre-merge adjustment using global delta statistics: "
                         "(1) norm-clamp delta where rel-L2 > 0.02 (caps overly-aggressive "
                         "sources, e.g. Crow-style attn deltas), "
                         "(2) sign-flip gate: weight *= exp(-100 * flip_rate) when flip_rate "
                         "> 0.01 (down-weights destructive sources per tensor), "
                         "(3) consensus bonus: when 2 sources' delta-cosine > 0.003, both "
                         "weights *= 1.2 (amplifies aligned merge gains, MLP middle layers). "
                         "Empirically motivated by delta_analysis_jackrong_vs_crow.csv. "
                         "Critical layers are excluded (handled by --pr682-turbo). Default OFF.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard-size", type=float, default=5.0,
                    help="Target output shard size in GB (default: 5)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="Output dtype for merged weights")
    ap.add_argument("--device", default=None,
                    help="Device for merge computation (cpu or cuda). Default: auto-detect (cuda if available, else cpu).")
    ap.add_argument("--no-gpu", action="store_true",
                    help="Force CPU-only merge even if GPU is available.")
    args = ap.parse_args()

    out_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]  # noqa: F841
    shard_bytes = int(args.shard_size * 1024**3)

    # Device selection: GPU by default if available
    if args.no_gpu:
        args.device = "cpu"
    elif args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device  : {args.device}", flush=True)

    if len(args.source) < 2:
        print("ERROR: need at least 2 sources for TIES merge", file=sys.stderr)
        sys.exit(1)

    # Parse weights
    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
        if len(weights) != len(args.source):
            print(f"ERROR: --weights has {len(weights)} values but --source has {len(args.source)}", file=sys.stderr)
            sys.exit(1)
    else:
        weights = [1.0 / len(args.source)] * len(args.source)

    args.output.mkdir(parents=True, exist_ok=True)

    # Parse skip-patterns: substrings that gate "copy from base instead of merge".
    skip_patterns: List[str] = []
    if args.skip_patterns:
        skip_patterns = [p.strip() for p in args.skip_patterns.split(",") if p.strip()]

    # Auto MLP-skip for Qwen3.6 family.
    # Qwen3.5 and Qwen3.6 share model_type="qwen3_5" so we can't distinguish via that.
    # Qwen3.6's config.json has output_gate_type="swish"; Qwen3.5's has it null/absent.
    # Both transformers and llama.cpp ignore the field at runtime, but it's a clean
    # propagated identifier (dare_ties_merge.py copies config.json from --base verbatim).
    # Qwen3.6's think-policy lives in a fragile MLP attractor — small (1-2% rel L2)
    # MLP perturbations flip the policy across the always-think boundary (see
    # memory/feedback_qwen3_6_merge_policy_fragility.md). Qwen3.5 is robust to the
    # same perturbation (Omnimerge-v2 published, 0.2% leak).
    auto_mlp = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    base_cfg_path = args.base / "config.json"
    base_gate_type = None
    if base_cfg_path.exists():
        try:
            base_gate_type = json.loads(base_cfg_path.read_text()).get("output_gate_type")
        except Exception as e:
            print(f"WARNING: could not parse {base_cfg_path}: {e}", flush=True)
    if base_gate_type is not None and not args.no_auto_mlp_skip:
        added = [p for p in auto_mlp if not any(p in s or s in p for s in skip_patterns)]
        if added:
            skip_patterns.extend(added)
            print(f"  auto MLP-skip ENABLED: base output_gate_type={base_gate_type!r} "
                  f"(Qwen3.6 family — fragile policy in MLP). Added {added} to skip_patterns.",
                  flush=True)
        else:
            print(f"  auto MLP-skip would activate (output_gate_type={base_gate_type!r}), "
                  f"but MLP patterns already present in --skip-patterns.", flush=True)
    elif base_gate_type is not None and args.no_auto_mlp_skip:
        print(f"  WARNING: base output_gate_type={base_gate_type!r} (Qwen3.6 family) but "
              f"--no-auto-mlp-skip given. MLP layers WILL be merged. Expect think-leak in output. "
              f"Use this only for ablation studies.", flush=True)
    elif base_gate_type is None:
        print("  auto MLP-skip not triggered: base output_gate_type is null/absent "
              "(Qwen3.5 family or non-qwen3_5 — MLP merging is safe).", flush=True)

    print("=== Model merge ===")
    print(f"  method  : {args.method}")
    print(f"  base    : {args.base}")
    for i, s in enumerate(args.source):
        print(f"  source {i}: {s}  (weight={weights[i]:.4f})")
    print(f"  output  : {args.output}")
    print(f"  density : {args.density}{'  (ignored for task_arithmetic)' if args.method == 'task_arithmetic' else ''}")
    if args.method == "della":
        print(f"  epsilon : {args.epsilon}")
    darex_q = args.darex_q
    v2_features = None
    if args.method == "omnimerge_v2":
        if darex_q is None:
            darex_q = args.density  # default: same as DARE
        v2_features = set(f.strip() for f in args.v2_features.split(",") if f.strip())
        print(f"  darex_q : {darex_q}")
        print(f"  features: {sorted(v2_features)}")
    print(f"  weights : {weights} (sum={sum(weights):.4f})")
    print(f"  dtype   : {args.dtype}")
    print(f"  seed    : {args.seed}")
    print(f"  shard   : {args.shard_size} GB")
    if skip_patterns:
        print(f"  skip    : {skip_patterns}  (matching tensors copied from base)")
    print(flush=True)

    print("Loading weight maps...", flush=True)
    base_wm, base_names = load_weight_map(args.base)
    src_wms = [load_weight_map(s)[0] for s in args.source]
    print(f"  base has {len(base_names)} tensors", flush=True)

    # Sanity: check coverage
    missing_in_sources = []  # noqa: F841
    for s_idx, wm in enumerate(src_wms):
        s_set = set(wm.keys())
        base_set = set(base_wm.keys())
        only_in_base = base_set - s_set
        only_in_src = s_set - base_set
        if only_in_src:
            print(f"  WARNING: source {s_idx} has {len(only_in_src)} tensors NOT in base (will be ignored)", flush=True)
        if only_in_base:
            print(f"  NOTE   : source {s_idx} missing {len(only_in_base)} tensors from base (will copy from base)", flush=True)

    print("Opening shard handles...", flush=True)
    base_handles = open_shard_handles(args.base, base_wm)
    src_handles_list = [open_shard_handles(s, wm) for s, wm in zip(args.source, src_wms)]
    print(f"  base: {len(base_handles)} shards", flush=True)
    for i, h in enumerate(src_handles_list):
        print(f"  source {i}: {len(h)} shards", flush=True)

    # Load Fisher scores if provided
    fisher_handles = None
    if args.fisher:
        fisher_paths = [Path(p.strip()) for p in args.fisher.split(",")]
        if len(fisher_paths) != len(args.source):
            print(f"ERROR: --fisher has {len(fisher_paths)} files but --source has {len(args.source)}", file=sys.stderr)
            sys.exit(1)
        fisher_handles = []
        for fp in fisher_paths:
            if not fp.exists():
                print(f"ERROR: Fisher file not found: {fp}", file=sys.stderr)
                sys.exit(1)
            fisher_handles.append(safe_open(fp, framework="pt", device="cpu"))
            print(f"  fisher: {fp} ({len(fisher_handles[-1].keys())} tensors)", flush=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    # Stream tensors in base order, write sharded output
    current_shard: Dict[str, torch.Tensor] = {}
    current_shard_bytes = 0
    shard_idx = 1
    all_shards: List[str] = []
    weight_map_out: Dict[str, str] = {}

    t_start = time.time()
    total_tensors = len(base_names)
    print(f"Starting tensor loop over {total_tensors} tensors...", flush=True)

    def drop_pagecache():
        """Flush filesystem pages + release reclaimable pagecache. Requires root
        (we are, inside the pod container). Prevents the mmap'd source shards'
        file-backed pages from accumulating against the cgroup memory limit."""
        try:
            os.sync()
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3\n")
        except Exception as e:
            print(f"  drop_pagecache failed: {e}", flush=True)

    def flush_shard(final: bool = False):
        nonlocal current_shard, current_shard_bytes, shard_idx
        if not current_shard:
            return
        placeholder_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        out_path = args.output / placeholder_name
        print(f"  [shard {shard_idx}] writing {len(current_shard)} tensors, {human_bytes(current_shard_bytes)} → {placeholder_name}", flush=True)
        save_file(current_shard, str(out_path), metadata={"format": "pt"})
        all_shards.append(placeholder_name)
        for name in current_shard:
            weight_map_out[name] = placeholder_name
        current_shard = {}
        current_shard_bytes = 0
        shard_idx += 1
        # Drop pagecache to prevent mmap'd source shards from bloating RSS
        drop_pagecache()

    DEBUG = os.environ.get("MERGE_DEBUG", "") == "1"
    for i, name in enumerate(base_names):
        if DEBUG or i < 5:
            print(f"  >> [{i}] {name}", flush=True)
        base_t = get_tensor(base_handles, base_wm, name)
        if base_t is None:
            continue
        if DEBUG or i < 5:
            print(f"     base loaded shape={tuple(base_t.shape)} dtype={base_t.dtype}", flush=True)

        # Skip-pattern: copy from base, no merging. Used to preserve fragile
        # weight regions on the target base (e.g. Qwen3.6 MLPs — see
        # feedback_qwen3_6_merge_policy_fragility.md).
        if skip_patterns and any(p in name for p in skip_patterns):
            if DEBUG or i < 5:
                print("     skip-pattern match → copying from base (no merge)", flush=True)
            merged = base_t
        # Integer tensors (buffers, indices) are just copied from base
        elif not base_t.is_floating_point():
            merged = base_t
        else:
            # Collect source tensors; skip sources that don't have this tensor
            sources = []
            for s_idx, (wm, handles) in enumerate(zip(src_wms, src_handles_list)):
                t = get_tensor(handles, wm, name)
                if t is None:
                    continue
                if t.shape != base_t.shape:
                    print(f"  WARNING: {name}: source {s_idx} shape {t.shape} != base {base_t.shape} — skipping source", flush=True)
                    continue
                sources.append(t)

            if len(sources) < 2:
                # Not enough sources for TIES consensus — copy base
                merged = base_t
            else:
                # Use only the weights corresponding to sources that were actually kept
                # (sources missing this tensor or shape-mismatched are dropped earlier).
                # For simplicity, if any source was dropped, renormalize the weights.
                if len(sources) == len(args.source):
                    tensor_weights = weights
                else:
                    # Rare fallback: should be very few tensors where this happens
                    tensor_weights = [1.0 / len(sources)] * len(sources)
                if DEBUG or i < 5:
                    nelem = base_t.nelement()
                    print(f"     merging {len(sources)} sources, nelem={nelem:,} ({nelem*4/1024**3:.2f} GB fp32)", flush=True)
                # Load Fisher scores for this tensor if available
                fisher_ts = None
                if fisher_handles:
                    fisher_ts = []
                    for fh in fisher_handles:
                        keys = fh.keys()
                        # Try exact name, or common prefix mappings
                        if name in keys:
                            fisher_ts.append(fh.get_tensor(name))
                        else:
                            # Fisher may have model.* prefix or not
                            fisher_ts = None
                            break

                try:
                    merged = dare_ties_merge_tensor(
                        base_t, sources, args.density, generator,
                        weights=tensor_weights, method=args.method, epsilon=args.epsilon,
                        darex_q=darex_q, device=args.device,
                        fisher_tensors=fisher_ts, v2_features=v2_features,
                        tensor_name=name,
                        pr682_turbo=args.pr682_turbo,
                        m7_detector=args.m7_detector,
                        m7_layer_aware=args.m7_layer_aware,
                        m7_tau_norm=args.m7_tau_norm,
                        m7_flip_strength=args.m7_flip_strength,
                    )
                except Exception as e:
                    print(f"     ERROR merging {name}: {type(e).__name__}: {e}", flush=True)
                    raise
                if DEBUG or i < 5:
                    print(f"     merged dtype={merged.dtype}", flush=True)

        current_shard[name] = merged.contiguous()
        current_shard_bytes += merged.element_size() * merged.nelement()

        if (i + 1) % 50 == 0 or (i + 1) == total_tensors:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total_tensors - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total_tensors}] {name[:60]:60s} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

        if current_shard_bytes >= shard_bytes:
            flush_shard()
            gc.collect()

    flush_shard(final=True)

    # Rename shards to include total shard count
    total_shards = len(all_shards)
    final_shards = []
    final_wm: Dict[str, str] = {}
    for i, placeholder in enumerate(all_shards):
        final = f"model-{i+1:05d}-of-{total_shards:05d}.safetensors"
        (args.output / placeholder).rename(args.output / final)
        final_shards.append(final)
    for name, ph in weight_map_out.items():
        idx = all_shards.index(ph)
        final_wm[name] = final_shards[idx]

    # Write index.json
    total_size = sum((args.output / s).stat().st_size for s in final_shards)
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": final_wm,
    }
    with open(args.output / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Copy non-weight files.  Config, chat template, and processor configs come
    # from the base.  The BPE tokenizer (tokenizer.json, vocab.json, merges.txt)
    # comes from the FIRST SOURCE — fine-tuned models may ship an updated
    # tokenizer with different BPE merge order or extra added tokens, and the
    # merged weights were trained against that tokenizer, not the base's.
    # tokenizer_config.json comes from base (has the full chat template) but is
    # patched below to include any extra added_tokens from the source tokenizer.
    from_base = [
        "config.json", "generation_config.json",
        "tokenizer_config.json",
        "chat_template.jinja", "chat_template.json",
        "processor_config.json", "preprocessor_config.json",
        "video_preprocessor_config.json", "image_processor_config.json",
    ]
    from_source = [
        "tokenizer.json", "vocab.json", "merges.txt",
        "special_tokens_map.json",
    ]
    for f in from_base:
        src = args.base / f
        if src.exists():
            shutil.copy2(src, args.output / f)
            print(f"  copied {f} (from base)")
    source_dir = args.source[0]
    for f in from_source:
        src = source_dir / f
        if src.exists():
            shutil.copy2(src, args.output / f)
            print(f"  copied {f} (from source)")
        elif (args.base / f).exists():
            shutil.copy2(args.base / f, args.output / f)
            print(f"  copied {f} (from base, source missing)")

    # Patch tokenizer_config.json: merge any extra added_tokens from the source
    # tokenizer into the base's added_tokens_decoder so the tokenizer recognizes
    # all tokens the source models were trained with.
    tc_path = args.output / "tokenizer_config.json"
    stok_path = source_dir / "tokenizer.json"
    if tc_path.exists() and stok_path.exists():
        tc = json.load(open(tc_path))
        stok = json.load(open(stok_path))
        base_ids = set(tc.get("added_tokens_decoder", {}).keys())
        for tok in stok.get("added_tokens", []):
            tid = str(tok["id"])
            if tid not in base_ids:
                tc.setdefault("added_tokens_decoder", {})[tid] = {
                    "content": tok["content"],
                    "lstrip": tok.get("lstrip", False),
                    "normalized": tok.get("normalized", False),
                    "rstrip": tok.get("rstrip", False),
                    "single_word": tok.get("single_word", False),
                    "special": tok.get("special", True),
                }
        with open(tc_path, "w") as f:
            json.dump(tc, f, indent=2, ensure_ascii=False)
        print(f"  patched tokenizer_config.json (added_tokens_decoder: {len(tc['added_tokens_decoder'])})")

    elapsed = time.time() - t_start
    print(f"\n=== DONE in {elapsed/60:.1f} min ===")
    print(f"  {total_shards} shards, {human_bytes(total_size)}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
