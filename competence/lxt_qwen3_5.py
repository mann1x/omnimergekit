"""
lxt.efficient adapter for the Qwen3.5 family (transformers `qwen3_5` arch).

Mirrors lxt/efficient/models/qwen3.py — re-uses the standard gated-MLP and
attention patches, ships custom RMSNorm forwards for the two Qwen3_5 RMSNorm
variants whose attribute conventions differ from the lxt default.

Differences from vanilla Qwen3:
  - Qwen3_5RMSNorm        stores eps as `self.eps` (not `variance_epsilon`)
                          AND applies `(1.0 + self.weight)` Gemma-style.
  - Qwen3_5RMSNormGated   takes an optional `gate` arg; multiplies the
                          normalized output by `SiLU(gate)` when given.
  - Qwen3_5GatedDeltaNet  linear-attention block. Deliberately NOT patched.
                          The default backward (gradient × input via PyTorch
                          autograd) gives a usable attribution and the
                          relevance accumulator above it still receives a
                          meaningful signal. Patching this with proper LRP
                          requires a custom rule for the gated-delta
                          recurrence and is out of scope here.

If lxt upstream merges this, drop into lxt/efficient/models/qwen3_5.py and
add the import to lxt/efficient/models/__init__.py.
"""
from functools import partial

import torch
from torch.nn import Dropout
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
)

from lxt.efficient.patches import (
    patch_method,
    gated_mlp_forward,
    cp_gated_mlp_forward,
    dropout_forward,
    wrap_attention_forward,
    check_already_patched,
)
from lxt.efficient.rules import stop_gradient


def qwen3_5_patch_attention(module):
    """
    Custom patch_attention for the qwen3_5 family.

    Newer transformers exposes ALL_ATTENTION_FUNCTIONS as an
    `AttentionInterface` (a dict subclass with a `get_interface(name, default)`
    method). lxt's stock patch_attention REPLACES that with a plain dict,
    which breaks the model's `.get_interface(...)` call site:

        AttributeError: 'dict' object has no attribute 'get_interface'

    For attn_implementation="eager" — the only mode we use here — the model
    looks up the attention impl with `get_interface("eager", default=...)`;
    "eager" is not a registered key, so it falls back to the `default` arg,
    which is the module-level `eager_attention_forward`. That means we only
    need to swap `eager_attention_forward` to get LRP-aware attention; we
    can leave `ALL_ATTENTION_FUNCTIONS` untouched.

    For sdpa/flash attn modes you'd also need to wrap each AttentionInterface
    entry in place (preserving the AttentionInterface type) — out of scope
    here. eval_implementation="eager" is the standard for LRP/Fisher work
    anyway because LRP-aware backward needs explicit attention weights.
    """
    new_forward = wrap_attention_forward(module.eager_attention_forward)
    if check_already_patched(module.eager_attention_forward, new_forward):
        return False
    module.eager_attention_forward = new_forward
    return True


def qwen3_5_patch_cp_attention(module):
    # Same logic for the cp variant; lxt's `patch_cp_attention` has the same
    # AttentionInterface-vs-dict bug.
    from lxt.efficient.patches import wrap_cp_attention_forward
    new_forward = wrap_cp_attention_forward(module.eager_attention_forward)
    if check_already_patched(module.eager_attention_forward, new_forward):
        return False
    module.eager_attention_forward = new_forward
    return True


def qwen3_5_rms_norm_forward(self, x):
    """
    LRP-aware forward for Qwen3_5RMSNorm.

    Same identity-rule trick as lxt's rms_norm_forward (stop gradient through
    the variance term), but uses `self.eps` and the Gemma-style
    `(1.0 + self.weight)` scale that Qwen3_5 inherits.
    """
    input_dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * stop_gradient(torch.rsqrt(variance + self.eps))
    out = x32 * (1.0 + self.weight.float())
    return out.type_as(x)


def qwen3_5_rms_norm_gated_forward(self, hidden_states, gate=None):
    """
    LRP-aware forward for Qwen3_5RMSNormGated.

    Variance term is stop-gradient'd. The optional `gate` factor is applied
    after the weight, mirroring the original implementation. SiLU(gate) is
    element-wise so it propagates relevance proportionally — no extra rule
    needed.
    """
    import torch.nn.functional as F
    input_dtype = hidden_states.dtype
    h32 = hidden_states.to(torch.float32)
    variance = h32.pow(2).mean(-1, keepdim=True)
    h32 = h32 * stop_gradient(torch.rsqrt(variance + self.variance_epsilon))
    out = self.weight * h32.to(input_dtype)
    if gate is not None:
        out = out * F.silu(gate.to(torch.float32)).to(input_dtype)
    return out.to(input_dtype)


attnLRP = {
    Qwen3_5MLP: partial(patch_method, gated_mlp_forward),
    Qwen3_5RMSNorm: partial(patch_method, qwen3_5_rms_norm_forward),
    Qwen3_5RMSNormGated: partial(patch_method, qwen3_5_rms_norm_gated_forward),
    Dropout: partial(patch_method, dropout_forward),
    modeling_qwen3_5: qwen3_5_patch_attention,
}

cp_LRP = {
    Qwen3_5MLP: partial(patch_method, cp_gated_mlp_forward),
    Qwen3_5RMSNorm: partial(patch_method, qwen3_5_rms_norm_forward),
    Qwen3_5RMSNormGated: partial(patch_method, qwen3_5_rms_norm_gated_forward),
    Dropout: partial(patch_method, dropout_forward),
    modeling_qwen3_5: qwen3_5_patch_cp_attention,
}
