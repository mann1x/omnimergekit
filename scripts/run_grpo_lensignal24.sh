#!/usr/bin/env bash
# POWERED length-signal run: 24 steps, sized against the effect it must detect.
#
# WHY 24 STEPS
# ------------
# The 8-step run (2026-09-09, grpo_lensignal) produced a null result with NO POWER:
#   post-warmup mean_length 2286 2171 1921 1952 2315 1991 2400
#   OLS slope +13.4 tok/step, se 39.7, t = +0.34   (|t|>2.57 needed at df=5)
# The between-step sd of 194 implies a per-completion sd of ~1551 tok at n=64/step,
# so the wobble was entirely rollout sampling noise. se of the slope falls as
# sigma / n_steps^1.5, so:
#     7 steps -> se 36.7  (detects +-92 tok/step ... useless)
#    24 steps -> se  5.7  (detects +-14 tok/step = ~343 tok over the run)
# 24 is the cheap point on that curve: ~375 s/step => ~2.6 h.
# [[feedback_a_trend_needs_power_before_it_needs_an_explanation]]
#
# WHY --smoke-rows-per-tier 18 AND NOT 6
# --------------------------------------
# Steps consume 64 completions each (bsz 1 x grad_accum 32 x 2 devices), and an epoch
# is rows*num_generations = rows*8 completions. At 6 rows/tier (18 prompts) an epoch is
# 2.25 steps, so 24 steps would be ~10.7 EPOCHS over 18 prompts and mean_length would
# measure memorisation of those 18, not a length policy. 18 rows/tier (54 prompts) keeps
# the run at ~3.5 epochs -- identical to the 8-step run. Tiers stay balanced (18 each;
# the pool holds 24/96/128), so the tier mix is unchanged. ONLY power changes.
#
# BUDGETS
# -------
# Uses the budgets re-measured BY the 8-step run (n_pass_lens 33/112/142), not the
# pre-run file (9/66/76). The lcb_exec/T P35 moved 2635 -> 1622 once it had 33 samples
# instead of 9 -- the THIN warning was right and 2635 was never a stable estimate.
# The 8-step run measured nothing, so there is no comparability to preserve.
#
# UNCHANGED FROM THE 8-STEP RUN, DELIBERATELY: reward shape, lambda per tier,
# BUDGET_QUANTILE 0.35, LR 1e-6, num_generations 8, --max-completion-len 8192.
# The clamp hypothesis (lenpen = min(nt/budget,1.0) flattens ~65% of passers by
# construction) is NOT tested here; this run only asks whether lengths move.
set -euo pipefail
cd /srv/ml/repos/omnimergekit

export TOKENIZERS_PARALLELISM=false
export VLLM_SKIP_P2P_CHECK=0
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is BANNED here: it uses the CUDA VMM
# API, breaks cudaIpcGetMemHandle and kills vLLM TP>1. The OOM message suggests it. No.

exec /srv/ml/envs/envs/omk-grpo/bin/torchrun --nproc_per_node 2 --standalone \
  scripts/train_grpo_efficiency.py \
  --model /mnt/sdc/v7rework/arms/Jprime-p3-bf16 \
  --pool eval/efficiency/grpo_pool_v3_budgeted.jsonl \
  --output /mnt/sdc/v7rework/grpo_lensignal24 \
  --smoke --smoke-rows-per-tier 18 --smoke-steps 24 \
  --budgets /mnt/sdc/v7rework/budgets_from_lensignal8.json \
  --max-completion-len 8192 \
  --measure-out /mnt/sdc/v7rework/grpo_lensignal24/measured.json
