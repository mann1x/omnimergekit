#!/usr/bin/env bash
# R9 run4 -- TWO-CHECKPOINT eval chain: dose-parity arm + early low-dose arm.
#
# WHY TWO ARMS AND NOT ONE
# ------------------------
# run1-run3 concluded "GEPO costs capability" from a DOSE SERIES, not a single point.
# A single run4 cell can only say "run4 is/isn't better than armJ"; it cannot separate
# "same failure mode" from "right direction, too much dose". Two points on the dose axis
# reconstruct the series that produced the original conclusion.
#
#   checkpoint-60   ~0.46e-4 summed decayed LR  -- early, low dose
#   checkpoint-218  ~1.63e-4                    -- DOSE PARITY with run2's entire run
#
# Step 219 is the exact parity step; saves land on even steps only (SAVE_STEPS=2), so
# 218 is the nearest save -- 1 step of 424 short, 0.2% of the epoch.
#
# ORDER IS DELIBERATE: ck218 first. It is the decision-relevant arm, so if the chain
# dies overnight the arm that answers the question is the one already banked.
#
# GATES ON SENTINELS, NOT EXIT CODES: each stage must print its own DONE marker into its
# own log. A build script that dies after emitting partial artifacts can still exit 0
# through a pipe. [[feedback_gate_needs_a_sentinel_not_an_exit_code]]
#
# Adapters, checkpoints and results are NEVER deleted; this script only reads and adds.
set -uo pipefail

WORK=/mnt/sdc/ml/brevity/gepo
HERE=$(dirname "$0")
LOG=$WORK/chain_gepo4_2ck.log
ts(){ date -u '+%F %T UTC'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

# --- REFUSE under a live trainer -------------------------------------------------
# The build merges LoRA weights; a trainer mid-save would be read torn.
if pgrep -f '[g]epo_brevity.py' >/dev/null 2>&1; then
  say "FATAL: gepo_brevity.py still running -- refusing to build under a live trainer"
  exit 8
fi

# --- REFUSE unless BOTH checkpoints are complete ---------------------------------
# 12 files is the full save set (adapter+optimizer+scheduler+rng x2+tokenizer+state).
# The predicate lives in gate_ck_complete.sh, NOT inline: behind the refusal above it
# could never be exercised, and it is polarity-tested there (7 arms, incl. two real
# checkpoints as positive controls and two mid-save truncations as negatives).
for ck in checkpoint-218 checkpoint-60; do
  out=$(bash "$HERE/gate_ck_complete.sh" "$WORK/run4/$ck" 2>&1) || {
    say "FATAL: $out -- refusing"; exit 9; }
  say "$out"
done

run_arm(){          # ck_dir_name  tag
  local CK=$1 TAG=$2
  local BL=$WORK/build_a3b_$TAG.log
  local SL=$WORK/suite_$TAG.log

  say "===== ARM $TAG : build from $CK ====="
  if grep -q "A3B_${TAG^^}_BUILD_DONE" "$BL" 2>/dev/null; then
    say "SKIP build $TAG (sentinel already present)"
  else
    ADAPTER=$WORK/run4/$CK bash "$HERE/build_a3b_gepo4.sh" 2>&1 | tee -a "$BL"
    if ! grep -q "A3B_${TAG^^}_BUILD_DONE" "$BL"; then
      say "FATAL: build $TAG produced no BUILD_DONE sentinel -- stopping chain"
      return 1
    fi
  fi
  say "BUILD_OK $TAG"

  # NOTE: suite_gepo4.sh does `exec >>"$LOG"` (line 61) -- it redirects its OWN stdout
  # into $WORK/suite_$TAG.log. Piping it to tee therefore captures NOTHING, and the
  # SUITE_..._DONE sentinel never reaches the piped file. Read its own log instead.
  say "===== ARM $TAG : suite (log: $SL) ====="
  if grep -q "SUITE_${TAG^^}_DONE" "$SL" 2>/dev/null; then
    say "SKIP suite $TAG (sentinel already present)"
  else
    TAG=$TAG bash "$HERE/suite_gepo4.sh"
    if ! grep -q "SUITE_${TAG^^}_DONE" "$SL" 2>/dev/null; then
      say "FATAL: suite $TAG produced no SUITE_DONE sentinel in $SL"
      return 1
    fi
  fi
  say "SUITE_OK $TAG"
  return 0
}

rc=0
run_arm checkpoint-218 gepo4ck218 || rc=1
# ck60 still runs even if ck218 failed: an independent arm is worth banking, and the
# failure above is reported either way.
run_arm checkpoint-60  gepo4ck60  || rc=1

# --- COMBINED TABLE: both new arms in ONE cohort table ---------------------------
# suite_gepo4.sh tables only its own TAG; the dose SERIES needs both columns together.
say "=== armJ vs gepo1/2/3 vs gepo4ck60 vs gepo4ck218 -- qwen_suite (sampler=recommended) ==="
/srv/ml/envs/envs/omnimergekit/bin/python \
    "$HERE/table_gepo_arms.py" "gepo4ck60=qwena3bgepo4ck60_q6k" "gepo4ck218=qwena3bgepo4ck218_q6k" \
    2>&1 | tee -a "$LOG"

echo "###### CHAIN_GEPO4_2CK_DONE rc=$rc $(ts) ######" | tee -a "$LOG"
exit $rc
