#!/usr/bin/env bash
# R9 run4 -- rebalanced mixed pool + reward v2. The run that answers the actual
# question, after run1-run3 all failed and the control that WORKED was found.
#
# THE CONTROL: WHY AN/GEMMA-4-E2B SUCCEEDED AND R9 DID NOT
# --------------------------------------------------------
# GEPO is not broken. an-finetune/simpo/train_gepo_v16.sh ran the same algorithm on
# Gemma-4-E2B and got exactly what was wanted: mean_length 779.8 -> 683.9 (-12.3%)
# with no score damage. R9 produced COMPARABLE brevity -- run2 moved in-run mean
# length 6593 -> 6031 (-8.5%), and the served LCB eval showed -12.7% tokens/answer --
# and still LOST capability. So brevity pressure is not what broke it. Four
# differences, and the pool and the LR are the ones that can explain a capability
# loss at constant length:
#
#   axis            AN (worked)                 R9 run1-3 (failed)
#   pool            894 rows, 3 sources         128 rows, LCB ONLY
#   LR              1e-6                        5e-6            (5x)
#   temperature     0.9                         0.6
#   len/budget      779/512 = 1.52 (steep)      6031/12288 = 0.49 (shallow)
#
# A 5x learning rate on a 7x smaller SINGLE-DOMAIN pool is a drift story, not a
# brevity story, and the measured evidence agrees: the benches that lost the most had
# FLAT completion lengths across the whole dose series.
#   bench        armJ -> gepo2     completion p50    thinking_est
#   GPQA         83.33 -> 75.76    804 -> 792        p50 = 0
#   HumanEval    98.17 -> 94.51    220 -> 223        max = 0
#   HumanEval+   89.63 -> 87.80    226 -> 220        max = 0
# Scores fell while lengths did not move. Whatever cost 7.58pp of GPQA, it was not
# the model being shorter. [[project_r9_gepo_router_arm_refuted]]
#
# WHAT CHANGES, AND WHY EACH ONE
# ------------------------------
# 1. POOL: one mixed 849-row pool (build_gepo_mixed_pool.py), matched to AN's 894.
#      lcb_exec  /think    128  lambda=0.7   <- unchanged from run2/3, so the brevity
#                                               signal is not diluted and the run2
#                                               comparison still holds
#      mbpp_exec /no-think 371  lambda=0
#      mbpp_exec /think    100  lambda=0     <- DISJOINT problems from the no-think
#                                               slice; the same problem in both is one
#                                               problem at double weight, not two
#      mc_letter /no-think 250  lambda=0
#    27% of rows carry thinking. BOTH modes are trained because neither uniform choice
#    is defensible: an all-thinking pool leaves mc_letter scoring ZERO for every
#    rollout (measured 2026-08-28: GPQA rollouts run 16-18k tokens and 2 of 4 never
#    answer inside 24576, and attempt 3's smoke had mc_letter 8/8 clipped), and an
#    all-no-think pool trains the model out of thinking altogether.
#    mc_letter is no-think ONLY -- 16-18k at G=8 does not fit 96 GB or the wall clock
#    at any budget, and half never answer regardless. That is not a departure from
#    what we score: the served GPQA eval already runs at thinking_est p50=0 with
#    completion p50 ~800 on EVERY arm.
#
# 2. REWARD v2: length scored GROUP-RELATIVELY among passing rollouts on a robust
#    scale-free statistic, s = clip((median-len)/max(MAD, 2%*median), -1, +1),
#    r = 0.6 + 0.4*s. v1 put ~11% of within-group variance on length, falling to 0.8%
#    as lengths shrank -- the signal DIED as the model converged. v2 measures 86.5%,
#    and still 86.5% at 10x shorter lengths. Truncated rollouts are CENSORED: excluded
#    from the median/MAD and scored strictly below the uncapped band, so "hit the cap"
#    can never be confused with "was merely longest".
#    [[feedback_cap_asymmetry_turns_a_bench_into_a_length_meter]]
#
# 3. LR 5e-6 -> 1e-6 and TEMPERATURE 0.6 -> 0.9, both taken from the AN control.
#
# WHAT IS DELIBERATELY UNCHANGED: lambda 0.7, budget 12288, beta 0.02, G=8, r/alpha
# 32/64, dropout 0.05, and ROUTER_LORA=0 -- run3 tested widening LoRA scope to the 40
# MoE routers and it LOST (suite mean 0.8882, lowest of all four arms; armJ->gepo3 on
# lcb_v6_77q -12.99pp, McNemar p=0.0213, the only one of six pairwise deltas in the
# cohort to clear the measured 5.81pp paired SE). Scope is not the lever.
#
# THIS RUN CHANGES THREE THINGS AT ONCE (pool, reward, LR+temp). That is deliberate
# and it is not an ablation: run1-3 already established that changing ONE thing at a
# time on a 128-row LCB-only pool loses. The question now is whether the configuration
# the AN control says should work, works. If it does, the ablation is worth 60h. If it
# does not, this line of attack is finished and no ablation would rescue it.
#
# GATES BEFORE THE 41h LEG
#   1. artifact contamination gate on the mixed pool, on THIS host
#   2. gate_learned      -- non-zero grad_norm AND reward_std
#   3. REPLAY_TIER_ALIVE -- all FOUR tiers must produce a non-zero reward. The smoke
#      draws one problem per tier (--limit-per-tier 1) at the FULL 12288 budget,
#      because a 2048-capped smoke starved the replay tiers into clipped=1.000/nz=0
#      and proved nothing (bug-643).
#
# WALL CLOCK: priced at 12.66M generated tokens / ~296k tok/h -- run2's MEASURED rate
# (1024 rollouts x mean_tok 5780 in 19h59m) -- = ~43h.
#
# STEP COUNT, corrected 2026-08-28. I first wrote "849/(1*16*2) = ~26 steps". That is
# wrong: TRL expands the dataset to ROLLOUTS, not problems, and num_iterations=2 runs
# two optimizer passes over each generation batch. The real count is
#     849 problems x G=8 = 6792 rollouts / (pdb 1 * accum 16 * world 2 = 32) x 2 = 424
# and the progress bar confirms 424. The wall clock is unaffected (it was priced from
# tokens, not steps) but the CHECKPOINT count is not: SAVE_STEPS=2 gives ~212 saves,
# and at a measured 537 MB each (run3/checkpoint-16) that is ~114 GB with
# save_total_limit=None. It fits /mnt/sdc (743 G free) and costs ~1.4% in save
# overhead, so run4 was left running rather than restarted -- but SAVE_STEPS should be
# ~20 (21 saves, ~11 GB) on the next run. Do NOT "fix" it to 8: on run2/3, whose epochs
# really were ~64 steps, 8 was already coarse, and the original reason for lowering it
# stands.
set -uo pipefail

REPO=/srv/ml/repos/omnimergekit

exec env \
  RUN=run4 \
  REWARD=v2 \
  POOL="$REPO/eval/replay/gepo_mixed_pool.jsonl" \
  POOL_LIMIT=0 \
  REPLAY_DIFFICULTY="${REPLAY_DIFFICULTY:-}" \
  ROUTER_LORA=0 \
  MAXCOMP=12288 \
  BUDGET=12288 \
  LAMBDA=0.7 \
  G=8 \
  ACCUM=16 \
  LR=1e-6 \
  TEMPERATURE=0.9 \
  BETA=0.02 \
  EPOCHS=1 \
  SAVE_STEPS=2 \
  bash "$(dirname "$0")/r9_gepo_hf.sh" "$@"
