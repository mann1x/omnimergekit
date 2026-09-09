#!/usr/bin/env bash
# Knowledge doc (READ FIRST, keep updated): docs/METHOD_grpo_efficiency.md
# AN-anchored brevity run for Jprime-p3 -- INTERRUPTIBLE and RESUMABLE by design.
#
#   START / RESUME :  bash scripts/run_grpo_an_anchored.sh
#   STOP CLEANLY   :  touch /mnt/sdc/v7rework/grpo_an_full/STOP
#                     (or: kill -USR1 <pid> -- the handler writes the same file)
#
# The run stops AT THE NEXT STEP BOUNDARY, writes a full checkpoint, and exits 0.
# Re-running this script picks up where it left off (--resume auto). Nothing is lost:
# weights, optimiser, scheduler, RNG and the step counter all travel in the checkpoint,
# and the script REFUSES to resume from one that is missing any of them.
#
# WHY THERE IS NO PILOT
# ---------------------
# A short pilot cannot see this effect, and that is measured, not assumed.
# an-finetune's grpo_v8_v2 is the only arm in this house that ever shortened
# significantly. Re-analysed as if its first 120 steps were a standalone pilot:
#     steps 1-120 : slope +0.298 tok/step  se 0.374  t=+0.80  FLAT  (+36 tok, UP)
#     steps 1-223 : slope -0.076            se 0.155  t=-0.49  FLAT
#     steps 1-446 : slope -0.129            se 0.053  t=-2.43  SHORTENS (-57 tok)
# Even first-30 vs last-30 over the FULL 446 steps is -44 tok (-7.3%), t=-1.18: not
# significant. The effect only emerges from an OLS over several hundred points, so the
# run is segmented rather than piloted -- stop it during the day, resume it overnight,
# and read the slope when the steps are in.
# [[feedback_a_trend_needs_power_before_it_needs_an_explanation]]
#
# SETTINGS, AND WHY EACH DIFFERS FROM THE 2026-09-09 lensignal24 ARM
# ------------------------------------------------------------------
#   --beta 0.01        lensignal24 inherited 0.04, justified in our own --beta help as
#                      "an-finetune used 0.04" -- that is AN's v1, whose length went
#                      NOWHERE (t=-0.36). The arm that worked ran 0.01. beta also bounds
#                      how far a KL excursion distorts the loss: lensignal24 step 20 hit
#                      kl=545 (0.04 -> ~21.8 of loss; 0.01 -> ~5.4).
#   --length-lambda 0.8  AN's winning value; our pool ships 0.1/0.1/0.3. This FLATTENS
#                      the per-tier weighting deliberately -- AN's design, not ours.
#   --budgets ..._an_anchored  budget = 0.82 x tier mean, AN's operating ratio.
#                      BUDGET_QUANTILE=0.35 put the driver tiers at 0.54-0.58x, i.e.
#                      MORE aggressive than the only configuration known to work.
#                      RAISES lcb 1622->2285 and mc_letter/T 1225->1875.
#   --max-completion-len 4096  AN's cap was 1.65x their mean (1024 over 620); ours pooled
#                      is 2028 -> AN-equivalent ~3350 -> 4096. MEASURED on this rig:
#                      cap 4096 -> 180 s/step, 4.7% clipped; cap 8192 -> 375 s/step,
#                      1.6% clipped. The step ends when the LONGEST rollout does, so the
#                      cap is ~a direct wall-clock multiplier. A 1024 cap would sit at
#                      0.37x the coding tier's typical answer and censor it out of
#                      existence, leaving no passing lengths to derive a budget from.
#                      A CAP CHANGE IS A BASIS CHANGE: every comparable arm uses 4096.
#   --max-grad-norm 1.0  what HF would default to, but STATED and logged. lensignal24
#                      clipped steps 20 (2.637) and 21 (1.135) silently.
#   --smoke-rows-per-tier 512  takes the WHOLE v4 pool (largest tier is 146, so 512 is
#                      simply "no cap"): 91 lcb_exec/T + 146 mc_letter/N + 128
#                      mc_letter/T = 365 rows. One accepted cost:
#                      (a) TIER BALANCE IS DELIBERATELY UNEQUAL -- 25/40/35%, not
#                          33/33/33, because the 60/40 driver/replay split is the
#                          design, not an accident. The POOLED mean_length is therefore
#                          a weighted average of three different jobs and must never be
#                          read as a trend on its own. READ THE PER-TIER `since=` ROWS
#                          AND THE PER-TIER `group classes` CENSUS -- a pooled census
#                          reported "clamp is the minor mode, 12.5%" on 2026-09-09 when
#                          the clamp was in fact flattening the entire brevity replay.
#                          [[feedback_batch_composition_is_an_eval_basis]]
#                          [[feedback_a_pool_wide_metric_cannot_see_a_minority_tier_objective]]
#                      (b) EPOCHS -- SOLVED, and not by growing the corpus. The fix
#                          was grad_accum 32 -> 16. AN converted data to optimiser
#                          updates at 32 completions/update (batch 8x4x1); at
#                          grad_accum 32 on 2 devices we were at 64 -- HALF AN's rate,
#                          so 2 epochs bought only 62 steps and 450 steps meant 14.5
#                          epochs. At 16, the 365-row v4 pool gives 91 steps/epoch, so
#                          150 steps = 1.64 epochs against AN's 2.00, in ~4 h.
#
#   pool v4 (365 rows)   driver 219 (60.0%) = manic-arm-contrast 128 + lcb_v6_easy 91
#                        replay 146 (40.0%) = gpqa_main_minus_diamond
#                        Built by scripts/build_grpo_pool_v4.py, which RE-DERIVES both
#                        holdouts from source: 0/198 GPQA-Diamond and 0/254 frozen-LCB
#                        eval ids present. manic-arm-contrast is EXHAUSTED at 128.
#
#   mask_truncated_completions=False   restores AN's actual brevity mechanism: a
#                        truncated rollout scores 0 and STAYS in the gradient. AN ran
#                        at clipped_ratio 0.53-0.69 that way. See train_grpo_efficiency.py.
set -euo pipefail
cd /srv/ml/repos/omnimergekit

# CEILING, not a target. lr_scheduler_type is constant_with_warmup -- NO decay -- so
# max_steps does not shape the LR trajectory and stopping early is LOSSLESS. Every step
# is checkpointed and STOP lands at the next boundary, so set the ceiling high and stop
# where you want rather than committing to a step count in advance.
#   150 steps = 1.64 epochs = ~3.5 h
#   250 steps = 2.75 epochs = ~5.9 h   <-- target: closest to AN's 2.00 that uses a night
#   300 steps = 3.30 epochs = ~7.1 h   <-- this ceiling
# at the measured 84.8 s/step (v4 smoke: 68-98 s over 6 steps, grad_accum 16, cap 4096).
STEPS="${STEPS:-300}"
OUT="${OUT:-/mnt/sdc/v7rework/grpo_an_short}"
BUDGETS="${BUDGETS:-/mnt/sdc/v7rework/budgets_an_anchored.json}"
ARCHIVE_EVERY="${ARCHIVE_EVERY:-10}"
SAVE_STEPS="${SAVE_STEPS:-1}"
KEEP="${KEEP:-3}"

[ -f "$BUDGETS" ] || { echo "REFUSE: budgets file $BUDGETS missing" >&2; exit 2; }
mkdir -p "$OUT"

# --- disk preflight -------------------------------------------------------------
# A checkpoint is adapter + optimiser + scheduler + RNG + tokenizer. Reference: an
# E2B r16 PEFT checkpoint is ~192 MB; this is r32 on a 26B MoE, so budget ~0.5 GB and
# verify against the FIRST real checkpoint rather than trusting this estimate.
NEED_GB=$(( (STEPS / ARCHIVE_EVERY + KEEP + 2) ))       # ~1 GB per retained checkpoint
FREE_GB=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc '0-9')
echo ">>> disk: ${FREE_GB}G free at $OUT, need ~${NEED_GB}G for $((STEPS / ARCHIVE_EVERY)) archives + $KEEP rolling"
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
  echo "REFUSE: only ${FREE_GB}G free, want ~${NEED_GB}G. Free space or lower ARCHIVE_EVERY." >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM=false
export VLLM_SKIP_P2P_CHECK=0
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is BANNED: the CUDA VMM API breaks
# cudaIpcGetMemHandle and kills vLLM TP>1. The OOM message suggests it. Do not.

SEG="$(date -u +%Y%m%dT%H%M%SZ)"
echo ">>> SEGMENT $SEG  steps=$STEPS  out=$OUT"
echo ">>> to stop cleanly:  touch $OUT/STOP"
exec /srv/ml/envs/envs/omk-grpo/bin/torchrun --nproc_per_node 2 --standalone \
  scripts/train_grpo_efficiency.py \
  --model /mnt/sdc/v7rework/arms/Jprime-p3-bf16 \
  --pool eval/efficiency/grpo_pool_v4_budgeted.jsonl \
  --output "$OUT" \
  --smoke --smoke-rows-per-tier 512 --smoke-steps "$STEPS" \
  --grad-accum 16 \
  --budgets "$BUDGETS" \
  --length-lambda 0.8 \
  --beta 0.01 \
  --max-grad-norm 1.0 \
  --budget-quantile 0.6 \
  --max-completion-len 4096 \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit "$KEEP" \
  --archive-every "$ARCHIVE_EVERY" \
  --archive-dir "$OUT/archive" \
  --resume auto \
  --stop-file "$OUT/STOP" \
  --measure-out "$OUT/measured_${SEG}.json" \
  --dump-rollouts "$OUT/rollouts_${SEG}"
