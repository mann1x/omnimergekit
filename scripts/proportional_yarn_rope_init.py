"""Register a `proportional_yarn` rope init function in transformers'
ROPE_INIT_FUNCTIONS, composing YaRN piecewise-frequency scaling over Gemma 4's
proportional rope base.

WHY THIS EXISTS — council audit (csl-2026-05-28-1825-5f1b, verdict BLOCKER):
transformers 5.5.0's `_compute_proportional_rope_parameters`
(modeling_rope_utils.py:187-254) reads ONLY `rope_theta`, `factor`,
`partial_rotary_factor` and applies simple `inv_freq /= factor` (line 253) —
plain Position Interpolation. It NEVER touches `yarn_factor`, `beta_fast`,
`beta_slow`, `mscale`, `mscale_all_dim`, or `original_max_position_embeddings`.

If you set `rope_parameters.full_attention.yarn_factor = 2.0` while leaving
`rope_type = 'proportional'`, transformers will silently apply PI — not YaRN.
The model trains on 256k positions reinterpreted as 512k via linear
interpolation only. NIAH / NoLiMa @ 512k would drop catastrophically and the
training run would burn 10+ hours before the eval gate detects it.

WHAT THIS MODULE DOES:
- Defines `_compute_proportional_yarn_parameters` per the YaRN paper
  (arXiv:2309.00071 §3.3, eqs. for ramp + base/theta blend) anchored on
  Gemma 4's proportional frequency base.
- Registers it as `ROPE_INIT_FUNCTIONS['proportional_yarn']`.
- Hooks `attention_factor` (the JSON key transformers reads) to 0.1·ln(s)+1
  when not explicitly set — per YaRN §5.1, mscale ≈ 1.0693 for s=2.0.

USAGE — import this module BEFORE any model load so the dispatch sees
'proportional_yarn' as a known key:

    import proportional_yarn_rope_init  # noqa: F401  (registration side-effect)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(yarn_cfg_dir, ...)

The matching `patch_yarn_config.py` writes:
    rope_parameters.full_attention.rope_type = 'proportional_yarn'
    rope_parameters.full_attention.yarn_factor = 2.0
    rope_parameters.full_attention.original_max_position_embeddings = 262144
    rope_parameters.full_attention.beta_fast = 32
    rope_parameters.full_attention.beta_slow = 1
    rope_parameters.full_attention.attention_factor = 1.0693    # NOT mscale!
    rope_parameters.full_attention.partial_rotary_factor = 0.25 # unchanged
    rope_parameters.full_attention.rope_theta = 1000000.0       # unchanged
"""
from __future__ import annotations

import math

import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


def _compute_proportional_yarn_parameters(config, device=None, seq_len=None,
                                          layer_type: str | None = None,
                                          head_dim_key: str = "head_dim"):
    """YaRN-over-proportional rope init for Gemma 4 full-attention layers.

    Signature matches transformers' rope-init contract: returns
    (inv_freq, attention_factor). attention_factor scales the softmax in
    Gemma4Attention; setting it to >1.0 partially recovers the entropy that
    longer-context softmax otherwise dilutes (the YaRN §5.1 attention-
    temperature correction).
    """
    # rope_parameters can be either a top-level dict or {layer_type: {...}}
    # depending on whether the model was built with per-layer rope configs
    # (Gemma 4 always uses the per-layer form).
    rope_params = (
        config.rope_parameters[layer_type]
        if layer_type and isinstance(config.rope_parameters, dict)
        and layer_type in config.rope_parameters
        else config.rope_parameters
    )

    # CRITICAL (verified 2026-06-07 against modeling_gemma4.py:1062, 1137):
    # Gemma 4's full-attention (global) layers run at global_head_dim=512, NOT
    # head_dim=256 (the sliding layers' value). transformers' own
    # Gemma4TextRotaryEmbedding injects head_dim_key='global_head_dim' ONLY when
    # `rope_type == 'proportional'` EXACTLY (line 1062). Our custom rope_type is
    # 'proportional_yarn', so that guard never fires and transformers calls us
    # WITHOUT head_dim_key — we'd default to 'head_dim'=256, computing rope_angles
    # =32 over a 128-dim rotated span when the global head is 512 (rope_angles must
    # be 64 over a 256-dim span). That mismatch silently corrupts the YaRN ramp →
    # NIAH/NoLiMa collapse at 512k after hours of training. Derive the right key
    # ourselves for full_attention layers.
    if layer_type == "full_attention" and getattr(config, "global_head_dim", None):
        head_dim_key = "global_head_dim"

    head_dim = getattr(config, head_dim_key, None)
    if head_dim is None:
        # Fallback: hidden_size // num_attention_heads (Gemma 4 has explicit
        # head_dim=256 / global_head_dim=512 in text_config, so this is defensive).
        head_dim = config.hidden_size // config.num_attention_heads

    base = float(rope_params["rope_theta"])
    factor = float(rope_params["factor"])
    rope_proportion = float(rope_params.get("partial_rotary_factor", 1.0))

    # Number of rotary frequency pairs = (rotated head dim) / 2.
    # Gemma 4 full_attention: global_head_dim=512, partial_rotary_factor=0.25 →
    # 128 rotated dims → 64 frequency pairs; remaining 384 dims pass through
    # unrotated (inv_freq length = 512//2 = 256: 64 real + 192 zero-pad).
    # (Sliding layers, head_dim=256 → 64 rotated → 32 pairs, but they use
    # rope_type='default' so this YaRN path is full_attention-only.)
    rope_angles = int(rope_proportion * head_dim // 2)

    # Step 1: proportional base frequencies — same as upstream's
    # _compute_proportional_rope_parameters, but in OUR control so we can
    # apply YaRN blending below. We build the inverse frequencies over the
    # ROTATED dim count (rope_angles*2), reading `rope_theta` directly.
    pos_freqs = base ** (
        torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32, device=device)
        / head_dim
    )

    # Step 2: YaRN ramp calculation.
    #
    # find_correction_dim(num_rotations, d, b, max_pos) gives the dim index at
    # which the rotation count over max_pos positions equals num_rotations.
    # YaRN passes through dims where this exceeds beta_fast (high-freq, local
    # tracking), linearly interpolates dims below beta_slow (low-freq, global
    # extrapolation), and blends the rest via a clamped ramp.
    dim = int(head_dim * rope_proportion)  # rotated dim count = 2 * rope_angles
    original_max = int(rope_params.get("original_max_position_embeddings", 262144))
    beta_fast = float(rope_params.get("beta_fast", 32))
    beta_slow = float(rope_params.get("beta_slow", 1))

    def find_correction_dim(num_rotations: float, d: int, b: float,
                            max_pos: int) -> float:
        return (d * math.log(max_pos / (num_rotations * 2 * math.pi))) \
            / (2 * math.log(b))

    low = max(0.0, find_correction_dim(beta_fast, dim, base, original_max))
    high = min(float(dim - 1),
               find_correction_dim(beta_slow, dim, base, original_max))

    # Ramp is built over the same length as pos_freqs (rope_angles entries).
    ramp = torch.clamp(
        (torch.arange(rope_angles, dtype=torch.float32, device=device) - low)
        / (high - low + 1e-3),
        0.0, 1.0,
    )

    # Step 3: blend interpolation (1/(factor * pos_freqs), ramp=0) with
    # extrapolation (1/pos_freqs, ramp=1). High-freq dims (ramp→1) pass
    # through; low-freq dims (ramp→0) get linearly interpolated.
    inv_freq_rotated = (1.0 / (factor * pos_freqs)) * (1.0 - ramp) \
        + (1.0 / pos_freqs) * ramp

    # Step 4: zero-pad for the un-rotated nope dims (partial_rotary_factor
    # < 1.0 means head_dim - rope_angles*2 dims have no rope at all).
    # transformers' downstream code expects inv_freq of length head_dim // 2
    # so we extend with zeros — the nope dims are effectively 0-frequency.
    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = torch.cat([
            inv_freq_rotated,
            torch.zeros(nope_angles, dtype=torch.float32, device=device),
        ], dim=0)
    else:
        inv_freq = inv_freq_rotated

    # Step 5: attention temperature (YaRN §5.1). If the user provided an
    # explicit `attention_factor`, honor it; otherwise compute the YaRN
    # default mscale = 0.1·ln(s) + 1 (≈ 1.0693 for s=2.0).
    attention_factor = rope_params.get("attention_factor")
    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0

    return inv_freq, float(attention_factor)


# Register at import time. Importing this module from a training/eval entry
# point is enough — no further wiring needed; transformers picks up the new
# key from the global ROPE_INIT_FUNCTIONS table during model __init__.
ROPE_INIT_FUNCTIONS["proportional_yarn"] = _compute_proportional_yarn_parameters
