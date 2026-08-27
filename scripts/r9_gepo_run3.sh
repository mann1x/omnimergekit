#!/usr/bin/env bash
# R9 run3 (#854) -- GEPO brevity on CoderX/armJ, THIRD pass: router in scope.
#
# ONE VARIABLE CHANGES vs run2: the LoRA scope gains the MoE router. Every other
# knob is byte-identical to r9_gepo_run2.sh, and the topology is the same
# CUDA_VISIBLE_DEVICES=0,1 DDP, so run3 is directly comparable to run2.
#
# WHY -- the 2026-08-26 scope audit
# ---------------------------------
# run2 trained 310 nn.Linear targets across 40/40 layers. Depth coverage was
# never the problem: the regex splits the hybrid correctly (self_attn on the 10
# full_attention layers, linear_attn on the 30 linear_attention layers). The
# problem is WIDTH. With moe_intermediate_size=512, top_k=8 and
# shared_expert_intermediate_size=512, the active FFN per token is
# 8*512 + 512 = 4608 wide, and run2 reached 512 of it -- 11.1%. The other 88.9%,
# plus the router that decides which experts fire at all, had no gradient.
#
# The measured outcome fits that shape exactly. gepo2 vs armJ, qwen_suite:
#   reasoning length  0.97x  (GPQA / IFEval / math500 p50) -- NO brevity bought
#   GPQA   -7.58pp, 25/10 discordant, McNemar p=0.0167, monotone with dose
#   HE     -3.66pp,  6/0  discordant,           p=0.0312, monotone with dose
# i.e. a policy perturbation with real capability cost and no movement on the
# objective -- what you would expect if the capacity governing "when to stop"
# is in the frozen part.
#
# WHY THE ROUTER AND NOT THE EXPERTS
# ----------------------------------
# Both are fused 3D nn.Parameter (Qwen3_5MoeExperts.gate_up_proj / down_proj,
# and mlp.gate.weight), so `target_modules` reaches neither -- that part of the
# old comment was correct. But peft 0.20.0's `target_parameters`
# (lora.ParamWrapper) reaches both. Measured on meta device at r=32:
#   experts  1,326,448,640 trainable = 5.18% of base = 31x run2  -> too expensive
#   router       2,856,960 trainable = 0.011% of base = run2/15  -> free
# The router exclusion also rested on "every router-only lever here has failed
# (T158, T196.SFT-3)" -- both are Gemma-4 router-KD runs against loop/rumination.
# Different architecture, objective and failure mode; never tested on Qwen3.6
# brevity-RL. That is what this run tests.
#
# THE CONFOUND, STATED UP FRONT
# -----------------------------
# lora.ParamWrapper refuses lora_dropout != 0, so run3 trains at dropout 0 while
# run1/run2 trained at 0.05. run3 therefore moves TWO things. Consequences:
#   * a POSITIVE run3 result is NOT cleanly attributable -- it needs the
#     dropout-0 base-scope control (SCOPE=base ROUTER_LORA=0 DROPOUT=0) before
#     "the router did it" can be claimed.
#   * a NULL run3 result needs no control: removing dropout can only increase
#     adaptation, so it cannot manufacture a null.
# Run the control only if run3 is positive. gepo_brevity.py records
# lora.router_lora / lora.dropout in run_meta.json so the two can never be
# silently tabled together.
#
# GATES (both fatal, both before the 18h leg)
#   1. gate_learned    -- non-zero grad_norm AND reward_std, as run1/run2.
#   2. gate_router_moved -- router lora_B off zero. gate_learned reads the TOTAL
#      grad_norm, which the 310 Linear targets make non-zero regardless, so it
#      cannot see a wrapped-but-dead router. LoRA B inits to exactly 0.
#
# WALL CLOCK: ~18-20h on BOTH GPUs, same as run2 (+0.011% trainable is noise).
set -uo pipefail

exec env \
  RUN=run3 \
  ROUTER_LORA=1 \
  POOL_LIMIT=128 \
  MAXCOMP=12288 \
  BUDGET=12288 \
  LAMBDA=0.7 \
  G=8 \
  ACCUM=16 \
  LR=5e-6 \
  BETA=0.02 \
  EPOCHS=1 \
  SAVE_STEPS=8 \
  bash "$(dirname "$0")/r9_gepo_hf.sh" "$@"
