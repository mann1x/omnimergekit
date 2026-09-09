#!/usr/bin/env bash
# Prove the checkpoint / STOP / resume path BEFORE committing nights to a 450-step run.
#
# Three claims are checked against artefacts, not exit codes:
#   1. a checkpoint contains EVERYTHING needed to resume (adapter+optimiser+scheduler+RNG)
#   2. touching STOP halts at the next step boundary, after a save, with rc=0
#   3. --resume auto continues from the saved global_step rather than restarting at 0
# Uses the REAL model and the REAL runner settings -- a validation on a toy config
# proves nothing about the run it is meant to protect.
set -uo pipefail
cd /srv/ml/repos/omnimergekit

OUT=/mnt/sdc/v7rework/grpo_resume_check
LOG=/mnt/sdc/v7rework/grpo_resume_check.log
rm -rf "$OUT"; mkdir -p "$OUT"
: > "$LOG"

common=(--model /mnt/sdc/v7rework/arms/Jprime-p3-bf16
        --pool eval/efficiency/grpo_pool_v3_budgeted.jsonl
        --output "$OUT"
        --smoke --smoke-rows-per-tier 128
        --budgets /mnt/sdc/v7rework/budgets_an_anchored.json
        --length-lambda 0.8 --beta 0.01 --max-grad-norm 1.0
        --max-completion-len 4096
        --save-steps 1 --save-total-limit 3 --archive-every 2
        --archive-dir "$OUT/archive" --stop-file "$OUT/STOP")

echo "=== PHASE 1: 3 steps, stop requested after the 2nd checkpoint appears" | tee -a "$LOG"
/srv/ml/envs/envs/omk-grpo/bin/torchrun --nproc_per_node 2 --standalone \
  scripts/train_grpo_efficiency.py "${common[@]}" --smoke-steps 3 \
  >> "$LOG" 2>&1 &
TRAIN_PID=$!
# Ask for a clean stop as soon as step 2 has been checkpointed.
( while kill -0 $TRAIN_PID 2>/dev/null; do
    [ -d "$OUT/checkpoint-2" ] && { echo ">>> harness: touching STOP" >> "$LOG"; \
                                    touch "$OUT/STOP"; break; }
    sleep 5
  done ) &
wait $TRAIN_PID; RC1=$?
echo "PHASE1_RC=$RC1" | tee -a "$LOG"

echo "=== artefact check" | tee -a "$LOG"
LAST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*-//' | sort -n | tail -1)
echo "last checkpoint: checkpoint-$LAST" | tee -a "$LOG"
for f in adapter_model.safetensors adapter_config.json optimizer.pt scheduler.pt trainer_state.json; do
  [ -f "$OUT/checkpoint-$LAST/$f" ] && echo "  OK   $f" | tee -a "$LOG" \
                                     || echo "  MISS $f" | tee -a "$LOG"
done
ls "$OUT/checkpoint-$LAST"/rng_state*.pth >/dev/null 2>&1 \
  && echo "  OK   rng_state*.pth" | tee -a "$LOG" || echo "  MISS rng_state*.pth" | tee -a "$LOG"
echo "archived: $(ls -d "$OUT"/archive/checkpoint-* 2>/dev/null | tr '\n' ' ')" | tee -a "$LOG"
du -sh "$OUT/checkpoint-$LAST" 2>/dev/null | tee -a "$LOG"

echo "=== PHASE 2: resume to 5 steps, must start from $LAST not 0" | tee -a "$LOG"
rm -f "$OUT/STOP"
/srv/ml/envs/envs/omk-grpo/bin/torchrun --nproc_per_node 2 --standalone \
  scripts/train_grpo_efficiency.py "${common[@]}" --smoke-steps 5 --resume auto \
  >> "$LOG" 2>&1
RC2=$?
echo "PHASE2_RC=$RC2" | tee -a "$LOG"
grep -aE "RESUMING from|STOP FILE|ARCHIVED|CHECKPOINT WARNING" "$LOG" | tail -10
echo "RESUME_VALIDATION_DONE rc1=$RC1 rc2=$RC2"
