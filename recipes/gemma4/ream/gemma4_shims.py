# Gemma-4 compatibility shims for the Samsung REAM merger.
#
# REAM is written against the Qwen MoE layout and assumes, everywhere:
#   * the decoder layer list is at   model.model.layers
#   * the MoE block is at            layer.mlp     (gate + experts in ONE module)
#   * one attention mask and one set of position embeddings serve EVERY layer
#
# Gemma-4 violates all three:
#   * layers live at                 model.model.language_model.layers
#   * layer.mlp is the DENSE always-on FFN (intermediate_size 2112); the MoE is
#     split across sibling modules  layer.router  and  layer.experts
#   * layer_types interleaves 25 'sliding_attention' with 5 'full_attention'
#     layers (idx 5/11/17/23/29 on the 26B-A4B), and the reference model selects
#     BOTH the mask and the position embeddings per layer type.
#
# The third point is the one that fails silently: feeding a sliding-window mask
# and the local RoPE to a global-attention layer still runs, still produces
# activations, and quietly computes the merge from corrupted hidden states.
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)


def resolve_text_model(model):
    """Return the Gemma4TextModel whose .layers REAM should walk.

    Accepts the full Gemma4ForConditionalGeneration (weights + vision + audio),
    the inner Gemma4Model, or an already-unwrapped text model. Raises rather
    than guessing: silently walking the wrong module list would merge nothing
    and still 'succeed'.
    """
    cand = model
    for path in (("model", "language_model"), ("language_model",), ("model",), ()):
        cur = cand
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok and hasattr(cur, "layers") and hasattr(cur, "embed_tokens"):
            if not hasattr(cur.config, "layer_types"):
                raise ValueError(
                    "resolved a module with .layers but its config has no "
                    "layer_types; this is not a Gemma-4 text model")
            return cur
    raise ValueError(
        f"could not resolve a Gemma-4 text model from {type(model).__name__}; "
        "expected .model.language_model / .language_model / .model with "
        ".layers and .embed_tokens")


class Gemma4GateShim(nn.Linear):
    """Presents Gemma4TextRouter to REAM as if it were Qwen's plain nn.Linear gate.

    REAM does `isinstance(moe_layer.gate, nn.Linear)` and then calls it, so this
    subclasses nn.Linear to keep that path live. It must NOT be a bare proj:
    Gemma4TextRouter.forward normalises and rescales before projecting, and
    skipping that would hand REAP saliency logits computed on raw hidden states.

    `weight` is the SAME Parameter object as router.proj.weight, so REAM's
    in-place `gate.weight.data = gate.weight.data[keep]` prunes the real router.
    """

    def __init__(self, router, raw_holder=None):
        n_exp, hidden = router.proj.weight.shape
        super().__init__(hidden, n_exp, bias=False)
        object.__setattr__(self, "_router", router)
        # Gemma-4 feeds the ROUTER the raw residual and the EXPERTS
        # pre_feedforward_layernorm_2(residual) -- two DIFFERENT tensors. Our hook
        # sits on .experts, so x here is already LN2'd. Gemma4MoEView captures the
        # raw residual into this holder; see forward().
        object.__setattr__(self, "_raw_holder",
                           raw_holder if raw_holder is not None else {})
        # share the Parameter object (not a copy) so REAM's slicing writes through
        del self._parameters["weight"]
        self._parameters["weight"] = router.proj.weight

    def forward(self, x):
        r = self._router
        # The real Gemma4TextDecoderLayer does:
        #     _, w, idx = self.router(hidden_states_flat)                    <- RAW residual
        #     hs2       = self.pre_feedforward_layernorm_2(hidden_states_flat)
        #     hs2       = self.experts(hs2, idx, w)                          <- LN2'd
        # Our hook target is .experts, so `x` is LN2(residual). Applying r.norm(x)
        # on top would compute proj(norm(LN2(residual))) instead of the router's
        # proj(norm(residual)) -- double-normalised gate logits, which corrupt REAP
        # saliency AND pseudo-group similarity, i.e. every merge decision downstream.
        src = self._raw_holder.get("residual")
        if src is None:
            raise RuntimeError(
                "Gemma4GateShim: raw residual was not captured. Refusing to fall back "
                "to the experts' LN2'd input -- that silently double-normalises the "
                "gate logits and corrupts the whole merge. Ensure Gemma4MoEView "
                "installed its pre_feedforward_layernorm_2 pre-hook.")
        if src.shape != x.shape:
            raise RuntimeError(
                f"Gemma4GateShim: captured residual {tuple(src.shape)} does not match "
                f"the experts' input {tuple(x.shape)} -- stale capture, refusing to guess.")
        h = r.norm(src)
        h = h * r.scale * r.scalar_root_size
        return F.linear(h, self.weight)

    # REAM prunes a router bias only under this GLM-era name. Gemma-4's
    # per_expert_scale is exactly that shape (E,) and MUST be pruned with the
    # expert set, or a 98-expert model keeps a 128-length scale vector.
    @property
    def e_score_correction_bias(self):
        return self._router.per_expert_scale

    @e_score_correction_bias.setter
    def e_score_correction_bias(self, value):
        if not isinstance(value, nn.Parameter):
            value = nn.Parameter(value.detach().clone())
        self._router.per_expert_scale = value

    def sync_out_features(self, n_experts):
        self.out_features = n_experts
        self._router.proj.out_features = n_experts
        self._router.config.num_experts = n_experts


class Gemma4MoEView:
    """A plain (non-nn.Module) view presenting layer.router + layer.experts as
    one Qwen-style MoE block.

    Deliberately NOT an nn.Module: REAM's write-back does `moe_layer.experts =
    <new experts>`, and on a Module that would register the new experts as a
    CHILD of the old one and serialise as `experts.experts.*`. As a plain object
    the setter writes straight through to the real decoder layer.
    """

    def __init__(self, layer, layer_ind, top_k):
        self.layer = layer
        # Capture the RAW residual -- pre_feedforward_layernorm_2's INPUT -- which is
        # exactly what the real router is fed. Without this the gate shim would see
        # the experts' already-normalised input. See Gemma4GateShim.forward().
        self._raw_holder = {}
        holder = self._raw_holder
        self._ln2_handle = layer.pre_feedforward_layernorm_2.register_forward_pre_hook(
            lambda _m, inputs: holder.__setitem__("residual", inputs[0]))
        self.gate = Gemma4GateShim(layer.router, holder)
        self.top_k = top_k
        self._layer_ind = layer_ind

    def release_hooks(self):
        h = getattr(self, "_ln2_handle", None)
        if h is not None:
            h.remove()
            self._ln2_handle = None

    @property
    def experts(self):
        return self.layer.experts

    @experts.setter
    def experts(self, value):
        self.layer.experts = value

    @property
    def num_experts(self):
        return self.layer.experts.num_experts

    @num_experts.setter
    def num_experts(self, value):
        self.layer.experts.num_experts = value
        self.gate.sync_out_features(value)

    def parameters(self, recurse=True):
        """The MoE block's parameters = router + experts.

        REAM calls num_parameters(moe_layer) to report per-layer compression;
        on Qwen that walks one Module. The view spans two, so yield both --
        omitting the router would under-report and, worse, hide a router that
        failed to prune alongside its experts.
        """
        yield from self.layer.router.parameters(recurse=recurse)
        yield from self.layer.experts.parameters(recurse=recurse)

    def to(self, *a, **kw):
        self.layer.to(*a, **kw)
        return self

    def __repr__(self):
        return (f"Gemma4MoEView(layer={self._layer_ind}, "
                f"experts={self.num_experts}, top_k={self.top_k})")


def attach_hook_view(layer, view):
    """Make layer.experts hookable by REAM's hook_fn, which receives the hooked
    module and reads `.gate` / `.experts` off it.

    Written into __dict__ so nn.Module.__setattr__ never sees them: the self
    reference would otherwise register the experts module as its own child and
    recurse through state_dict()/parameters().
    """
    layer.experts.__dict__["gate"] = view.gate
    layer.experts.__dict__["experts"] = layer.experts   # self-ref, unregistered
    layer.experts.__dict__["top_k"] = view.top_k
    layer.experts.__dict__["_layer_ind"] = view._layer_ind


def build_moe_views(text_model, top_k):
    views = []
    for i, layer in enumerate(text_model.layers):
        if not getattr(layer, "enable_moe_block", False):
            raise ValueError(
                f"layer {i} has no MoE block; this port assumes every Gemma-4 "
                f"layer is MoE (enable_moe_block=True on the 26B-A4B)")
        v = Gemma4MoEView(layer, i, top_k)
        attach_hook_view(layer, v)
        views.append(v)
    return views


def run_all_experts_gemma4(moe_layer, hidden_states, **kw):
    """REAM's run_all_experts unpacks (B, S, H).

    Gemma4TextDecoderLayer calls self.experts(hidden_states_2, ...) with the
    tokens ALREADY flattened to (B*S, H), so the hook sees a 2-D tensor. Fold
    it to (1, B*S, H): run_all_experts flattens internally anyway, so the
    expert math is identical and gate comes back 3-D as the merger asserts.
    """
    from ream.moe_utils import run_all_experts
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0)
    elif hidden_states.dim() != 3:
        raise ValueError(f"unexpected MoE input rank {hidden_states.dim()}")
    return run_all_experts(moe_layer, hidden_states, **kw)


def get_moe_input_gemma4(text_model, device, input_ids, attention_mask):
    """Gemma-4 replacement for ream.moe_utils.get_moe_input.

    Returns BOTH mask variants and BOTH position-embedding variants, keyed by
    layer type, instead of the single mask/RoPE the Qwen path assumes.
    """
    text_model.embed_tokens.to(device)
    inputs_embeds = text_model.embed_tokens(input_ids)

    cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)

    import inspect as _inspect
    _emb_kw = ("inputs_embeds"
               if "inputs_embeds" in _inspect.signature(create_causal_mask).parameters
               else "input_embeds")
    mask_kwargs = dict(
        config=text_model.config,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )
    mask_kwargs[_emb_kw] = inputs_embeds
    present = set(text_model.config.layer_types)
    builders = {
        "full_attention": create_causal_mask,
        "sliding_attention": create_sliding_window_causal_mask,
    }
    unknown = present - set(builders)
    if unknown:
        raise ValueError(f"unhandled Gemma-4 layer_types {sorted(unknown)}; "
                         "refusing to guess a mask for them")
    causal_mask_mapping = {lt: builders[lt](**mask_kwargs) for lt in present}

    # Gemma-4 keeps a SEPARATE inv_freq buffer per layer type, and its forward
    # requires the type: rotary_emb(x, position_ids, layer_type). Reusing one
    # layer type's RoPE for the other is silent -- it produces finite garbage.
    text_model.rotary_emb.to(device)
    pos_emb = {lt: text_model.rotary_emb(inputs_embeds, position_ids, lt) for lt in present}

    return {
        "hidden_states": inputs_embeds,
        "position_embeddings": pos_emb,
        "attention_mask": causal_mask_mapping,
        "position_ids": position_ids,
        "past_key_values": None,
        "use_cache": False,
        "cache_position": cache_position,
        "layer_types": list(text_model.config.layer_types),
    }


def moe_forward_gemma4(decoder_layer, inputs, layer_ind, i=None, chunk_size=None, device=None):
    """Gemma-4 replacement for ream.moe_utils.moe_forward: picks the mask and
    the position embeddings that belong to THIS layer's type."""
    if device is not None:
        decoder_layer.to(device)
    lt = inputs["layer_types"][layer_ind]
    mask = inputs["attention_mask"][lt]
    pos = inputs["position_embeddings"][lt]
    hs = inputs["hidden_states"] if i is None else inputs["hidden_states"][i:i + chunk_size]
    if mask is not None and i is not None:
        mask = mask[i:i + chunk_size]
    return decoder_layer(
        hidden_states=hs,
        position_embeddings=pos,
        attention_mask=mask,
        position_ids=inputs["position_ids"],
        past_key_values=None,
        use_cache=False,
        cache_position=inputs["cache_position"],
    )
