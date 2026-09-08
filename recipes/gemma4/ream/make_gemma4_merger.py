"""Generate gemma4_merger.py from the upstream REAM merger by asserted patch.

Every replacement asserts its anchor. If Samsung's merger.py changes upstream,
this fails loudly instead of silently emitting a half-ported file.
"""
import sys

SRC = "/shared/dev/ream/ream/merger.py"
DST = "/shared/dev/omnimergekit/recipes/gemma4/ream/gemma4_merger.py"

s = open(SRC).read()
UPSTREAM_LINES = len(s.splitlines())
applied = []


def rep(old, new, count=1, label=""):
    global s
    n = s.count(old)
    assert n == count, f"anchor {label!r}: expected {count} occurrence(s), found {n}"
    s = s.replace(old, new)
    applied.append(f"{label} ({count})")


# 1. relative -> absolute imports (file lives outside the ream package)
rep("from .", "from ream.", count=8, label="relative imports")

# 2/3. first_moe_layer + the Qwen top_k discovery block
rep(
    """        self.first_moe_layer = getattr(model.config, 'first_k_dense_replace', 0)  # e.g. for GLM models
        first_moe = model.model.layers[self.first_moe_layer].mlp
        if not hasattr(first_moe, 'top_k'):
            for idx in range(self.first_moe_layer, len(model.model.layers)):
                moe = model.model.layers[idx].mlp
                moe.top_k = moe.gate.top_k
                moe.num_experts = get_num_experts(moe)
""",
    """        # Gemma-4: every decoder layer carries a MoE block (enable_moe_block=True),
        # so there is no dense prefix to skip.
        self.first_moe_layer = 0
        self._tm = resolve_text_model(model)
        text_cfg = self._tm.config
        # Gemma-4 splits the MoE across sibling modules (layer.router / layer.experts)
        # and uses layer.mlp for the DENSE always-on FFN. The views present that pair
        # to REAM as one Qwen-style block; see gemma4_shims.Gemma4MoEView.
        self._views = build_moe_views(self._tm, text_cfg.top_k_experts)
        first_moe = self._views[self.first_moe_layer]
""",
    label="init: layers/views")

# 4. top_k assertion: Gemma-4 names it top_k_experts, not num_experts_per_tok
rep(
    "        assert self.top_k == model.config.num_experts_per_tok, (self.top_k, model.config.num_experts_per_tok)",
    "        assert self.top_k == text_cfg.top_k_experts, (self.top_k, text_cfg.top_k_experts)",
    label="top_k assert")

# 5. hook target: the real experts module, augmented by attach_hook_view
rep(
    "            mlp = (self.mtp_layer.layer if is_mtp else self.model.model.layers[layer_ind]).mlp",
    "            mlp = self._tm.layers[layer_ind].experts  # gate/experts attached by attach_hook_view",
    label="hook target")

# 6. per-layer-type mask + RoPE
rep(
    """                hid_states = moe_forward(self.model.model.layers[layer_ind],
                                         states,
                                         i=i_,
                                         chunk_size=self.batch_size,
                                         device=self.device).data""",
    """                hid_states = moe_forward_gemma4(self._tm.layers[layer_ind],
                                                states,
                                                layer_ind=layer_ind,
                                                i=i_,
                                                chunk_size=self.batch_size,
                                                device=self.device).data""",
    label="moe_forward -> gemma4")

rep("        states = get_moe_input(self.model, self.device, **self.batch)",
    "        states = get_moe_input_gemma4(self._tm, self.device, **self.batch)",
    label="get_moe_input -> gemma4")

# the mask is now a per-layer-type dict, so .shape no longer exists
rep("                  states['hidden_states'].shape, states['attention_mask'].shape)",
    "                  states['hidden_states'].shape,\n"
    "                  {k: (None if v is None else tuple(v.shape)) for k, v in states['attention_mask'].items()})",
    label="verbose mask print")

# 7. expert-count debug
rep(
    """                n_exp = get_num_experts(self.model.model.layers[layer_ind].mlp) if hasattr(
                    self.model.model.layers[layer_ind].mlp, 'experts') and not is_mtp else \\
                    (get_num_experts(self.mtp_layer.layer.mlp) if is_mtp else 'dense')""",
    "                n_exp = get_num_experts(self._views[layer_ind])",
    label="n_exp debug")

# 8. the per-layer MoE handle used by fit()
rep(
    "            moe_layer = (self.mtp_layer.layer if is_mtp else self.model.model.layers[layer_ind]).mlp  # expert ffn",
    "            moe_layer = self._views[layer_ind]  # Gemma4MoEView over router+experts",
    label="fit moe_layer")

# 9. debug prints of the merged blocks
rep("""            print(self.model.model.layers[self.first_moe_layer].mlp, flush=True)
            print(self.model.model.layers[self.first_moe_layer + 1].mlp, flush=True)
            print(self.model.model.layers[-1].mlp, flush=True)""",
    """            print(self._tm.layers[self.first_moe_layer].experts, flush=True)
            print(self._tm.layers[self.first_moe_layer + 1].experts, flush=True)
            print(self._tm.layers[-1].experts, flush=True)""",
    label="debug prints")

# 10. config write-back lands on the TEXT config
rep("""        moe_layer = self.model.model.layers[self.first_moe_layer].mlp
        self.model.config.num_experts = moe_layer.num_experts
        self.model.config.num_experts_per_tok = moe_layer.top_k""",
    """        moe_layer = self._views[self.first_moe_layer]
        self._tm.config.num_experts = moe_layer.num_experts
        self._tm.config.top_k_experts = moe_layer.top_k""",
    label="config write-back")

# 10b. fit()'s per-layer tally walks layer.mlp directly
rep("""            total_experts += get_num_experts(layer.mlp)
            top_k.append(layer.mlp.top_k)""",
    """            total_experts += get_num_experts(self._views[layer_ind])
            top_k.append(self._views[layer_ind].top_k)""",
    label="fit tally")

# 10c. the merged-layer debug line. get_num_experts takes the view; num_parameters
#      needs a real nn.Module, so it gets the experts module itself.
rep("""                  'num experts merge', get_num_experts(self.mtp_layer.layer.mlp
                                           if is_mtp else self.model.model.layers[layer_ind].mlp),""",
    """                  'num experts merge', get_num_experts(self._views[layer_ind]),""",
    label="debug num experts")
rep("""                  'num params merged', num_parameters(self.mtp_layer.layer.mlp
                                                      if is_mtp else self.model.model.layers[layer_ind].mlp),""",
    """                  'num params merged', num_parameters(self._tm.layers[layer_ind].experts),""",
    label="debug num params")

# 10d. Gemma-4 hands the experts module a FLATTENED (B*S, H) tensor
rep("                gate, final, act = run_all_experts(module,",
    "                gate, final, act = run_all_experts_gemma4(module,",
    label="run_all_experts -> gemma4")

# 11. everything remaining
n_left = s.count("self.model.model.layers")
assert n_left > 0
s = s.replace("self.model.model.layers", "self._tm.layers")
applied.append(f"residual self.model.model.layers ({n_left})")
if "self.model.config" in s:
    k = s.count("self.model.config")
    s = s.replace("self.model.config", "self._tm.config")
    applied.append(f"residual self.model.config ({k})")

HEADER = '''# VENDORED + PORTED from Samsung REAM `ream/merger.py` (see LICENSE in
# /shared/dev/ream). Generated by scripts/make_gemma4_merger.py -- do not hand-edit;
# change the generator so the port stays reproducible against upstream.
#
# Ported for Gemma-4 (gemma4_text): 30 layers, 128 experts, top_k_experts=8,
# hidden 2816 / dense FFN 2112 / expert 704, layer_types = 25 sliding + 5 full
# attention (idx 5/11/17/23/29 on the 26B-A4B).
#
# What upstream assumes and Gemma-4 breaks:
#   layers at model.model.layers          -> model.model.language_model.layers
#   MoE at layer.mlp                      -> layer.router + layer.experts siblings
#                                            (layer.mlp is the DENSE always-on FFN)
#   config.num_experts_per_tok            -> config.top_k_experts
#   one attention mask + one RoPE for all -> per-layer-type mask AND RoPE
#
# The last one is the dangerous one: it does not raise. A sliding-window mask on a
# full_attention layer still yields activations, so the merge would be computed
# from silently corrupted hidden states on 5 of 30 layers.
#
# Router pruning: Gemma4TextRouter carries per_expert_scale (E,) alongside
# proj.weight (E,H). BOTH must be sliced to the kept expert set. The shim exposes
# per_expert_scale under REAM's `e_score_correction_bias` name so upstream's
# existing pruning branch handles it.
'''
IMPORTS = ("from gemma4_shims import (resolve_text_model, build_moe_views,\n"
           "                          get_moe_input_gemma4, moe_forward_gemma4,\n"
           "                          run_all_experts_gemma4)\n")

lines = s.split("\n")
for i, ln in enumerate(lines):
    if ln.startswith("from ream.saliency import"):
        lines.insert(i + 1, IMPORTS.rstrip("\n"))
        break
else:
    sys.exit("could not find import anchor for gemma4_shims")
s = HEADER + "\n" + "\n".join(lines)

open(DST, "w").write(s)
import ast
ast.parse(s)
print(f"upstream merger.py: {UPSTREAM_LINES} lines")
print(f"wrote {DST}: {len(s.splitlines())} lines, syntax OK")
print("replacements applied:")
for a in applied:
    print("   -", a)
