"""End-to-end gate for the ported merger on a TINY synthetic Gemma-4.

Runs the real Merger.fit() -- views, hooks, per-layer-type mask/RoPE, grouping,
merging, router pruning, config write-back -- on a randomly initialised model
small enough to run on CPU in seconds. Scale is the only thing this does not
test, so it catches plumbing bugs before a multi-hour arm.
"""
import sys, os, json, tempfile, torch

sys.path.insert(0, "/shared/dev/ream")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

E_IN, E_OUT, L = 16, 8, 6
work = tempfile.mkdtemp(prefix="gemma4_merge_e2e_")
os.makedirs(os.path.join(work, "data"), exist_ok=True)
os.chdir(work)

cfg = Gemma4TextConfig(
    vocab_size=512, hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
    num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    num_experts=E_IN, top_k_experts=4, enable_moe_block=True,
    sliding_window=8, hidden_size_per_layer_input=0,
    layer_types=["sliding_attention"] * (L - 1) + ["full_attention"],
)
torch.manual_seed(0)
model = Gemma4ForCausalLM(cfg).eval()
print(f"synthetic model: {L} layers, {E_IN} experts, layer_types={cfg.layer_types}")
print(f"  full_attention layers: {[i for i,t in enumerate(cfg.layer_types) if t=='full_attention']}")

B, S = 32, 16
batch = {"input_ids": torch.randint(0, cfg.vocab_size, (B, S)),
         "attention_mask": torch.ones(B, S, dtype=torch.long)}
bf = f"data/tiny_b{B}_seq{S}_gemma4_seed42.pt"
torch.save(batch, bf)

from gemma4_merger import Merger  # noqa: E402

merger = Merger(model=model, merge_size=E_OUT, grouping="ream",
                merging="logits+weights", saliency="reap", dataset="tiny",
                mix_ratio="1.0", tokenizer_name="gemma4", batch_size=8,
                group_size=4, sequential=True, calibration_data_size=B,
                calibration_data_seq_len=S, seed=42, verbose=False)
merged = merger.fit()

fail = 0
tm = merger._tm
for i, layer in enumerate(tm.layers):
    ne = layer.experts.num_experts
    gu, dp = layer.experts.gate_up_proj.shape, layer.experts.down_proj.shape
    pw, pes = layer.router.proj.weight.shape, layer.router.per_expert_scale.shape
    bad = (ne != E_OUT or gu[0] != E_OUT or dp[0] != E_OUT
           or pw[0] != E_OUT or pes[0] != E_OUT)
    if bad or i == 0:
        print(f"  L{i}: experts={ne} gate_up={tuple(gu)} down={tuple(dp)} "
              f"proj={tuple(pw)} per_expert_scale={tuple(pes)}{'  <-- BAD' if bad else ''}")
    fail += bad
print(f"[1] all {L} layers pruned to {E_OUT}: {'FAIL' if fail else 'OK'}")

cfg_ok = merged.config.num_experts == E_OUT
print(f"[2] config.num_experts = {merged.config.num_experts} (expect {E_OUT}): "
      f"{'OK' if cfg_ok else 'FAIL'}")
fail += (not cfg_ok)

# no nested experts.experts.* leaked into the state dict
sd = merged.state_dict()
nested = [k for k in sd if ".experts.experts." in k]
print(f"[3] nested 'experts.experts.*' keys: {len(nested)} (expect 0): "
      f"{'OK' if not nested else 'FAIL ' + str(nested[:3])}")
fail += bool(nested)

with torch.no_grad():
    out = merged(input_ids=batch["input_ids"][:2], attention_mask=batch["attention_mask"][:2])
finite = torch.isfinite(out.logits).all().item()
print(f"[4] merged forward: logits {tuple(out.logits.shape)} all-finite={finite}: "
      f"{'OK' if finite else 'FAIL'}")
fail += (not finite)

print("\nRESULT:", "MERGER_E2E_OK" if fail == 0 else f"FAILURES={fail}")
sys.exit(1 if fail else 0)
