#!/usr/bin/env bash
# R9 run2 (#847) -- GEPO brevity on CoderX/armJ, second pass.
#
# Delta from run1, and why each number is what it is:
#
#   max_completion  8192 -> 12288   (== length_budget)
#   n_problems        64 -> 128     (--limit; rows[:128] is a SUPERSET of run1's
#                                    rows[:64], so run1's draw is contained)
#   optimizer steps   32 -> 64      (consequence, not a flag -- see below)
#
# WHY 12288 IS THE RIGHT CAP, NOT JUST A BIGGER ONE
# -------------------------------------------------
# The reward is correctness-gated and graded:
#     reward = (1 - lambda * min(ntok / budget, 1))   if the LCB tests pass
#            = 0.0                                     otherwise
# With lambda=0.7 and budget=12288, the penalty term spans [0, 0.7]. run1 capped
# at 8192, so its ratio could only reach 8192/12288 = 0.667 -- passing rewards
# lived in [0.533, 1.0] and the top third of the penalty range was structurally
# unreachable. Setting max_completion == length_budget opens it to [0.3, 1.0];
# min(.,1) saturates cleanly at the top, so there is no dead branch.
#
# The second, larger effect is on WHICH signal dominates. A clipped rollout emits
# unparsable code, fails the tests, and scores 0 -- a termination penalty, not a
# brevity gradient. run1's clipped_ratio ran 0.06-0.53 (mean 0.346) on EVERY one
# of its 16 generation rounds, so roughly a third of the run was training on that
# penalty instead of on the length term. 12288 moves most of those rollouts back
# under the ceiling.
#
# WHY 64 STEPS
# ------------
# Step count is not a flag -- it falls out of pool size, epochs, and the fixed
# cadence. Measured from run1's full.log: 32 optimizer steps / 16 generation
# rounds over 64 problems = 2 problems per optimizer step, 4 per round. 128
# problems x 1 epoch at the same cadence = 64 steps / 32 rounds, one clean pass
# with no repeated problems.
#
# 0.5 or 0.75 epoch (32 / 48 steps) are the cheaper options. Rejected: run1's
# mean_length never settled -- 5699 5706 4629 4142 4034 6342 4820 5035 5666 4994
# 5103 7305 5395 5289 4704 4370 across its 16 rounds, spiking at round 12. That
# is not a converged length policy, and halving the epoch while doubling the data
# would confound "more data" with "less optimization".
#
# WALL CLOCK: ~16-20h, BOTH GPUs
# ------------------------------
# run1: 6h14m55s for 64 problems = 349 s/problem at cap 8192; ~1140 s generation
# + ~257 s optimizer per round. A generation round ends when the SLOWEST of its
# sequences ends, and ~35% of them sat at the ceiling, so round time tracks the
# cap rather than the mean: 1140 * (12288/8192) ~ 1710 s generation + ~330 s
# optimizer = ~2040 s/round x 32 rounds ~ 18h. This blocks GPU0 and GPU1 for the
# duration, exactly as run1 did.
#
# UNCHANGED: length_budget 12288, lambda 0.7, num_generations 8, grad_accum 16,
# beta 0.02, lr 5e-6, seed 0, LoRA r32/alpha64. save_steps 8 -> 8 checkpoints
# over 64 steps (run1 saved 3 and only the last was ever evaluated).
#
# prompt 4096 + completion 12288 = 16384, under the 24576 server_max_model_len
# default, so that needs no change.
set -uo pipefail

exec env \
  RUN=run2 \
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
