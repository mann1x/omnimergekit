#!/bin/bash
# Smoke the NEW config: v4 pool (365 rows), grad_accum 16, mask_truncated=False.
# Proves four things the old config cannot:
#   1. pool v4 loads all 365 rows, tiers 91/146/128
#   2. grad_accum 16 -> 32 completions/step (was 64)
#   3. mask_truncated_completions=False -> truncated rollouts stay in the gradient
#   4. the PER-TIER group-classes census prints (pooled census is gone)
set -uo pipefail
cd /srv/ml/repos/omnimergekit
OUT=/mnt/sdc/v7rework/grpo_v4_smoke
rm -rf "$OUT"; mkdir -p "$OUT"
/srv/ml/envs/envs/omk-grpo/bin/torchrun --nproc_per_node 2 --standalone \
  scripts/train_grpo_efficiency.py \
  --model /mnt/sdc/v7rework/arms/Jprime-p3-bf16 \
  --pool eval/efficiency/grpo_pool_v4_budgeted.jsonl \
  --output "$OUT" \
  --smoke --smoke-rows-per-tier 512 --smoke-steps 6 \
  --grad-accum 16 \
  --budgets /mnt/sdc/v7rework/budgets_an_anchored.json \
  --length-lambda 0.8 --beta 0.01 --max-grad-norm 1.0 \
  --budget-quantile 0.6 --max-completion-len 4096 \
  --save-steps 1 --save-total-limit 2 \
  --archive-every 3 --archive-dir "$OUT/archive" \
  --stop-file "$OUT/STOP" \
  --measure-out "$OUT/measured.json" \
  --dump-rollouts "$OUT/rollouts"
echo "V4_SMOKE_RC=$?"
echo "V4_SMOKE_DONE"
