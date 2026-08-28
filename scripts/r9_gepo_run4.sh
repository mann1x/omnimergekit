#!/usr/bin/env bash
# R9 run4 -- new reward (v2) + capability replay. The two changes run1-run3 could not make.
#
# WHY run4 EXISTS: run1-3 ALL FAILED, AND THE REASON IS NOW MEASURED
# ------------------------------------------------------------------
# The v1 reward was  r = (1 - 0.7*min(ntok/12288,1))  if tests pass  else 0. GRPO
# normalises advantage by the WITHIN-GROUP std, so what matters is not the reward
# scale but the share of group variance each term owns. On a realistic group of 8
# with 5 passers:
#     total group std     0.3160
#     std among passers   0.0356   <- the ENTIRE length signal
#     length share        11.3%
# ~89% of the gradient said "be correct", ~11% said "be short" -- and that 11% fell
# to 0.8% when the same group's lengths were 10x smaller, i.e. the signal DIED as the
# model converged. Measured consequences across three runs:
#   run2  mean length moved 4% over a full epoch; reward rose mostly because fewer
#         rollouts were being zeroed by truncation (cliff-avoidance, not concision)
#   run3  adding the 40 MoE routers to LoRA scope changed nothing for the better:
#         suite MEAN 0.8882 (LOWEST of armJ/gepo1/gepo2/gepo3), and armJ->gepo3 on
#         lcb_v6_77q = -12.99pp, 13/3 discordant, McNemar p=0.0213 -- the ONLY one of
#         six pairwise deltas in the cohort to clear the measured 5.81pp paired SE.
#         Widening scope is not the lever. [[project_r9_gepo_router_arm_refuted]]
#
# WHAT CHANGES (two things, because you asked for both)
# -----------------------------------------------------
# 1. REWARD v2 -- length scored GROUP-RELATIVELY among passing rollouts, on a robust
#    scale-free statistic: s = clip((median-len)/max(MAD, 2%*median), -1, +1),
#    r = 0.6 + 0.4*s. Gate-measured length share of within-group variance: 86.5%,
#    and it STAYS 86.5% at 10x shorter lengths where v1 collapsed to 0.8%.
#    Invariants preserved and asserted: worst passer (0.2) still beats best failure
#    (0.0); a truncated rollout is CENSORED -- excluded from the median/MAD and scored
#    strictly below the uncapped band (0.1) so "hit the cap" can never again be
#    confused with "was merely longest".
# 2a. REPLAY IS RENDERED NO-THINK (REPLAY_NO_THINK=1). Measured 2026-08-28: with
#    thinking ON, GPQA replay rollouts run 16-18k tokens, 2 of 4 never reach an answer
#    inside 24576, and the tier scores ZERO for every rollout -- attempt 3's smoke had
#    mc_letter 8/8 clipped at the 12288 budget. Raising the cap is not available: ~18k
#    is needed, half never answer at any budget, the trainer already sat at 83/92 GB of
#    ~96 GB at G=8, and a G=8 GPQA group costs ~45 min (32 of them = ~24 h of a 25-30 h
#    run). With thinking OFF the same prompts answer 12/12 at p50 1595 tokens, 9/12
#    correct -- fits the existing budget, ~10x cheaper, and gives the correct/incorrect
#    mix GRPO needs.
#    This is NOT a departure from what we score: the served GPQA eval already runs at
#    thinking_est p50=0 with completion p50 ~800 on EVERY arm (armJ 83.33 / gepo1 79.80
#    / gepo2 75.76 / gepo3 76.77), and completion length is FLAT across that dose series
#    (804 -> 785) while the score falls 7.6pp -- so the GPQA loss is not a length effect
#    and the tier is defending accuracy, which is exactly what a lambda=0 correctness
#    gate trains. [[project_gpqa_cot_needs_16k_on_armj]]
#    LCB rows are unchanged: they render byte-identical to before.
#
# 2. REPLAY -- 64 lambda=0 problems mixed in, EQUAL split across the two tiers
#    (32 mc_letter from GPQA-main-minus-Diamond, 32 mbpp_exec from MBPP-minus-test).
#    Pure correctness, no length term: replay defends capability, and putting length
#    pressure on the reasoning GEPO already erodes is exactly backwards. The pool ratio
#    (35/65) is the INVERSE of the loss ratio it defends (GPQA -6.57pp vs HE -3.05pp),
#    hence --replay-balance equal rather than a proportional slice.
#
# SCOPE GOES BACK TO run2's. --router-lora is OFF: run3 tested it and it lost. That
# also restores lora_dropout 0.05, so run4 vs run2 differs in reward and replay only.
#
# TWO VARIABLES, BUT THEY ARE SEPARABLE BY WHICH METRIC MOVES:
#   brevity (LCB tok/answer, clipped_ratio) is what the REWARD is supposed to fix
#   capability (GPQA, HumanEval vs armJ)   is what REPLAY is supposed to hold
# A result where brevity improves and capability holds needs no ablation to read. A
# result where only one moves points at which change did it. Only a result where both
# move the WRONG way is ambiguous, and that outcome would end this line of attack
# anyway.
#
# GATES BEFORE THE LONG LEG
#   1. gate_learned      -- non-zero grad_norm AND reward_std (as run1-3)
#   2. REPLAY_TIER_ALIVE -- the smoke must show BOTH tiers producing reward, not just
#      LCB. An unwired tier scores 0.0 for every rollout: no within-group spread, no
#      gradient, and it drags the policy while the loss curve looks perfectly normal.
#      gate_learned cannot see this -- the LCB tier alone keeps grad_norm non-zero.
#
# WALL CLOCK: 128 LCB + 64 replay = 192 problems vs run2/3's 128, so ~48 generation
# steps instead of 32. LCB groups dominate cost (~1900s each); MBPP groups are much
# cheaper and GPQA sits between. Estimate 25-30h, i.e. LONGER than run3's 20h23m.
set -uo pipefail

REPO=/srv/ml/repos/omnimergekit

exec env \
  RUN=run4 \
  REWARD=v2 \
  REPLAY_POOL="$REPO/eval/replay/gepo_replay_pool.jsonl" \
  REPLAY_N=64 \
  REPLAY_BALANCE=equal \
  REPLAY_NO_THINK=1 \
  REPLAY_DIFFICULTY="$REPO/eval/replay/gepo_replay_pool.DIFFICULTY.json" \
  ROUTER_LORA=0 \
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
