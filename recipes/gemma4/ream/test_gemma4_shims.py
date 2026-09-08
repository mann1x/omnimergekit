"""Correctness gate for the Gemma-4 REAM port.

Runs on randomly-initialised modules built from the real text config -- no 49 G
load. What it proves:
  1. Gemma4GateShim reproduces Gemma4TextRouter's expert scores EXACTLY.
     (A bare router.proj on raw hidden states does NOT -- that is the silent
     failure this port exists to avoid, so it is asserted to differ.)
  2. REAM's run_all_experts drives Gemma4TextExperts' stacked layout and
     matches a hand-rolled reference expert forward.
  3. Pruning through the shim slices proj.weight AND per_expert_scale together.
"""
import sys, json, torch
import torch.nn.functional as F

sys.path.insert(0, "/shared/dev/ream")
sys.path.insert(0, "/shared/dev/omnimergekit/recipes/gemma4/ream")

from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextRouter, Gemma4TextExperts)
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from ream.moe_utils import run_all_experts, get_num_experts
from gemma4_shims import Gemma4GateShim

torch.manual_seed(0)
CFG = json.load(open(sys.argv[1]))["text_config"]
cfg = Gemma4TextConfig(**{k: v for k, v in CFG.items() if k != "architectures"})
# shrink only what does not change the math under test
cfg.num_experts, cfg.top_k_experts = 16, 4
cfg.hidden_size, cfg.moe_intermediate_size = 64, 32
E, H = cfg.num_experts, cfg.hidden_size

router = Gemma4TextRouter(cfg).eval()
with torch.no_grad():
    router.proj.weight.normal_(0, 0.02)
    router.scale.normal_(1.0, 0.05)
    router.per_expert_scale.normal_(1.0, 0.05)
experts = Gemma4TextExperts(cfg).eval()
with torch.no_grad():
    experts.gate_up_proj.normal_(0, 0.02)
    experts.down_proj.normal_(0, 0.02)

B, S = 2, 7
x = torch.randn(B, S, H)
flat = x.view(B * S, H)
fail = 0

# --- 1. gate shim vs the real router -----------------------------------------
shim = Gemma4GateShim(router).eval()
with torch.no_grad():
    ref_probs, _, _ = router(flat)          # softmax(expert_scores)
    shim_logits = shim(flat)
    shim_probs = F.softmax(shim_logits, dim=-1)
    naive = F.linear(flat, router.proj.weight)   # the WRONG port
    naive_probs = F.softmax(naive, dim=-1)

d_shim = (shim_probs - ref_probs).abs().max().item()
d_naive = (naive_probs - ref_probs).abs().max().item()
print(f"[1] shim vs router  max|dprob| = {d_shim:.3e}")
print(f"[1] naive proj      max|dprob| = {d_naive:.3e}   (must be LARGE)")
if d_shim > 1e-6:
    print("    FAIL: shim does not reproduce the router"); fail += 1
if d_naive < 1e-4:
    print("    FAIL: naive path indistinguishable -- test is not sensitive"); fail += 1

# --- 2. run_all_experts drives the stacked layout ----------------------------
shim2 = Gemma4GateShim(router).eval()
experts.__dict__["gate"] = shim2
experts.__dict__["experts"] = experts
print(f"[2] get_num_experts = {get_num_experts(experts)} (expect {E})")
if get_num_experts(experts) != E:
    print("    FAIL: expert count"); fail += 1
with torch.no_grad():
    logits, outs, acts = run_all_experts(experts, x, gated_sim=False)
    i = 3
    g, u = F.linear(flat, experts.gate_up_proj[i]).chunk(2, dim=-1)
    ref = F.linear(experts.act_fn(g) * u, experts.down_proj[i])
d_exp = (outs[i] - ref).abs().max().item()
print(f"[2] expert[{i}] vs reference max|d| = {d_exp:.3e}   shapes {tuple(logits.shape)} {tuple(outs.shape)}")
if d_exp > 1e-5:
    print("    FAIL: run_all_experts disagrees with reference expert"); fail += 1

# --- 3. pruning slices proj.weight AND per_expert_scale ----------------------
keep = list(range(0, E, 2))
with torch.no_grad():
    shim.weight.data = shim.weight.data[keep]
    shim.e_score_correction_bias = shim.e_score_correction_bias[keep]
    shim.sync_out_features(len(keep))
ok_w = tuple(router.proj.weight.shape) == (len(keep), H)
ok_s = tuple(router.per_expert_scale.shape) == (len(keep),)
print(f"[3] after prune: proj.weight {tuple(router.proj.weight.shape)} "
      f"per_expert_scale {tuple(router.per_expert_scale.shape)} (expect {(len(keep), H)} / {(len(keep),)})")
if not (ok_w and ok_s):
    print("    FAIL: router pruning did not write through"); fail += 1

print("\nRESULT:", "PORT_SHIMS_OK" if fail == 0 else f"FAILURES={fail}")
sys.exit(1 if fail else 0)
