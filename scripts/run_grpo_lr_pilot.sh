#!/usr/bin/env bash
# GRPO LR pilot -- 30 steps, ONE variable, gated on KL DIRECTION.
#
# WHY THIS EXISTS
#   grpo_an_short (2026-09-10) ran 300 steps / 7h59m and returned a null: completion
#   length t=-0.41, reward t=+0.39. The reward was NOT the problem -- within-group
#   Spearman(ntok, reward) was -0.34 to -0.86 with 76-97% of live groups correctly
#   signed, so the gradient pointed toward SHORTER the whole time.
#
#   What was wrong: run_grpo_an_anchored.sh never passes --lr, so the argparse default
#   1e-6 applied (train_grpo_efficiency.py:159). With median grad_norm 0.107 that is a
#   per-step update of ~1e-7. The confirming signature is the KL DIRECTION: a policy
#   that is actually moving away from its reference has RISING KL. Ours FELL, 0.0199 ->
#   0.0027 -- at lr=1e-6 the reward gradient could not overcome even a beta=0.01 anchor.
#
#   The launcher's comment block justifies beta, budget quantile, completion cap,
#   max-grad-norm and mask_truncated; GRPOConfig pins other TRL defaults with the note
#   "explicit so a default change is loud". learning_rate -- the one parameter that
#   scales EVERY update -- was the one left implicit. So: STATE IT.
#
# ONE VARIABLE. Penalty shape and the lcb_exec/T G 8->16 fix are both DEFERRED. They
# are defensible on their own terms but changing them here would confound the LR test,
# and the offline rollout re-scoring (scripts/rescore_grpo_penalty_shapes.py) can
# evaluate them for free afterwards.
#
# THE GATE IS KL DIRECTION, NOT REWARD AND NOT STD.
#   - reward has no power at 30 steps (the 300-step run's own reward t was +0.39)
#   - within-group std is NOT a health metric: the budget sweep showed std and
#     live-group share rise monotonically as the budget is loosened, because at a loose
#     budget ~everything sits under it and the budget stops being a target
#   - KL direction answers the one question this pilot exists to ask: is the policy
#     moving at all?
set -euo pipefail
cd /srv/ml/repos/omnimergekit

LR="${LR:?LR must be set explicitly -- the whole point of this pilot. e.g. LR=1e-5}"
STEPS="${STEPS:-30}"
OUT="${OUT:-/mnt/sdc/v7rework/grpo_lr_pilot_lr${LR}}"
BUDGETS="${BUDGETS:-/mnt/sdc/v7rework/budgets_an_anchored.json}"

[ -f "$BUDGETS" ] || { echo "REFUSE: budgets file $BUDGETS missing" >&2; exit 2; }
# Never overwrite a previous arm's results. [[feedback_adapters_and_results_are_never_deleted]]
[ -e "$OUT" ] && { echo "REFUSE: $OUT already exists -- pick a new OUT, never overwrite an arm" >&2; exit 2; }
mkdir -p "$OUT"

FREE_GB=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -lt 20 ] && { echo "REFUSE: only ${FREE_GB}G free at $OUT" >&2; exit 2; }

export TOKENIZERS_PARALLELISM=false
export VLLM_SKIP_P2P_CHECK=0
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is BANNED (breaks cudaIpcGetMemHandle
# and kills vLLM TP>1). The OOM message suggests it. Do not.

SEG="$(date -u +%Y%m%dT%H%M%SZ)"
echo ">>> LR PILOT  lr=$LR  steps=$STEPS  out=$OUT  seg=$SEG"
echo ">>> gate: KL must be RISING by step 25. Check with:"
echo ">>>   python scripts/grpo_kl_gate.py $OUT"
echo ">>> to stop cleanly:  touch $OUT/STOP"
exec /srv/ml/envs/envs/omk-grpo/bin/torchrun --nproc_per_node 2 --standalone \
  scripts/train_grpo_efficiency.py \
  --model /mnt/sdc/v7rework/arms/Jprime-p3-bf16 \
  --pool eval/efficiency/grpo_pool_v4_budgeted.jsonl \
  --output "$OUT" \
  --lr "$LR" \
  --num-generations "${GENS:-8}" \
  --lora-r 32 --lora-alpha 64 --seed "${SEED:-3407}" \
  --smoke --smoke-rows-per-tier 512 --smoke-steps "$STEPS" \
  --grad-accum 16 \
  --budgets "$BUDGETS" \
  --length-lambda 0.8 \
  --beta 0.01 \
  --max-grad-norm 1.0 \
  --budget-quantile 0.6 \
  --max-completion-len 4096 \
  --save-steps 5 \
  --save-total-limit 3 \
  --archive-every 10 \
  --archive-dir "$OUT/archive" \
  --resume auto \
  --stop-file "$OUT/STOP" \
  --measure-out "$OUT/measured_${SEG}.json" \
  --dump-rollouts "$OUT/rollouts_${SEG}"
