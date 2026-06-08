#!/bin/bash
# run_longctx_512k.sh — top-level cascade orchestrator for T87.
#
# Drives: phase0 anchors (solidpc) → phase1 train (blackswan-2 GPU 0+1) →
# phase2 eval (blackswan-2) → cascade gate (31B success → v6-coder go).
#
# Default: --dry-run mode prints the full plan, touches nothing. Pass --run
# to launch. Pass --target 31b|v6c to limit to one target.
#
# ### COUNCIL — read this script header before approving. The execution
# order, gate definitions, and abort signals all live here.
#
# Cascade order:
#   1. 31B-it          (dense, simpler, validates recipe)
#   2. 98e-v6-coder-it (MoE-pruned, harder, cascade-gated on #1)
#
# Per target:
#   a. Phase 0 — solidpc anchors (~2-4h, no blackswan-2 burn)
#   b. Phase 1 — blackswan-2 training (~6-14h, dual-GPU)
#   c. Phase 2 — blackswan-2 eval (~4-6h)
#
# All artifacts under:
#   solidpc:    backup_models/eval_results_longctx_512k_anchors/<target>/
#               backup_models/runs/longctx_512k/<target>/
#   blackswan2: /srv/ml/runs/longctx_512k/<target>/
#               /srv/ml/eval_results_longctx_512k/<target>/

set -uo pipefail

POD=${POD:-linode-blackswan-2}
DO_RUN=0
ONLY_TARGET=""
SKIP_PHASE0=0
SKIP_PHASE1=0
SKIP_PHASE2=0
while [ $# -gt 0 ]; do
  case "$1" in
    --run)           DO_RUN=1; shift;;
    --pod)           POD=$2; shift 2;;
    --target)        ONLY_TARGET=$2; shift 2;;
    --skip-phase0)   SKIP_PHASE0=1; shift;;
    --skip-phase1)   SKIP_PHASE1=1; shift;;
    --skip-phase2)   SKIP_PHASE2=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

# Target → friendly tag mapping
declare -A T_TAG=( [31b]="gemma-4-31B-it" [v6c]="gemma-4-A4B-98e-v6-coder-it" )
declare -A T_PUBLISHED=( [31b]="google/gemma-4-31B-it" [v6c]="mannix/gemma-4-A4B-98e-v6-coder-it" )

TARGETS=(31b v6c)
if [ -n "$ONLY_TARGET" ]; then
  TARGETS=("$ONLY_TARGET")
fi

echo "=================== T87 LONG-CONTEXT 512k CASCADE ==================="
echo "  targets       : ${TARGETS[*]}"
echo "  POD           : $POD"
echo "  do-run        : $DO_RUN  (0 = dry-run, 1 = execute)"
echo "  skip-phase0   : $SKIP_PHASE0"
echo "  skip-phase1   : $SKIP_PHASE1"
echo "  skip-phase2   : $SKIP_PHASE2"
echo
echo "Per target, this orchestrator will:"
echo "  Phase 0 (solidpc, no pod burn): canonical 9-bench@32k + RULER NIAH@{32k,64k,128k,256k} + MRCR-v2 8-needle@128k"
echo "  Phase 1 (blackswan-2):          YaRN-2.0 + LoRA continued pretrain ~250-300M tokens; GPU 0 trainer, GPU 1 probe watcher"
echo "  Phase 2 (blackswan-2):          merge LoRA → 9-bench@32k + NIAH@{32k…512k} + MRCR@256k + (v6c only) routing-entropy probe"
echo "  Gate after phase 2:             see gemma4_512k_plan_v2.md §Gate (pass/caveat-publish/halt-cascade)"
echo
echo "Cascade rule: if first target (31B) hits any HARD-FAIL gate, halt before launching v6-coder."
echo "================================================================"

if [ "$DO_RUN" -ne 1 ]; then
  echo "[dry-run] nothing executed. Re-run with --run after council approves."
  exit 0
fi

# Real execution path
for t in "${TARGETS[@]}"; do
  tag=${T_TAG[$t]}
  echo
  echo "================ TARGET: $t ($tag) ================"

  # -------- Phase 0 --------
  if [ "$SKIP_PHASE0" -ne 1 ]; then
    echo "[phase0] anchors on solidpc — ~2-4h, no pod burn"
    bash "$(dirname "$0")/phase0_anchor.sh" --target "$t" --run
    if [ $? -ne 0 ]; then
      echo "FATAL: phase0 failed on $t — halting cascade"
      exit 3
    fi
  else
    echo "[phase0] SKIPPED"
  fi

  # -------- Phase 1 --------
  if [ "$SKIP_PHASE1" -ne 1 ]; then
    echo "[phase1] training on $POD GPU 0 + probe on GPU 1 — ~6-14h"
    ssh -o BatchMode=yes "$POD" "bash /srv/ml/scripts/phase1_train.sh --target $t --run"
    # phase1_train.sh launches the training + probe in tmux; this ssh call
    # returns immediately after launch. The orchestrator polls for completion:
    while ssh -o BatchMode=yes "$POD" "test -f /srv/ml/runs/longctx_512k/$t/train.pid && kill -0 \$(cat /srv/ml/runs/longctx_512k/$t/train.pid) 2>/dev/null"; do
      sleep 600
      echo "  $(date -u +%H:%M:%SZ) training still running on $POD"
    done
    echo "[phase1] training finished on $t"
  else
    echo "[phase1] SKIPPED"
  fi

  # -------- Phase 2 --------
  if [ "$SKIP_PHASE2" -ne 1 ]; then
    echo "[phase2] eval on $POD — ~4-6h"
    ssh -o BatchMode=yes "$POD" "bash /srv/ml/scripts/phase2_eval.sh --target $t --run"
  else
    echo "[phase2] SKIPPED"
  fi

  # -------- Cascade gate (between targets) --------
  if [ "$t" = "31b" ] && [ ${#TARGETS[@]} -gt 1 ]; then
    echo
    echo "[cascade gate] checking 31B Phase 2 verdict before launching v6-coder"
    # phase2_eval.sh writes a verdict file: pass / caveat / hard-fail
    verdict=$(ssh -o BatchMode=yes "$POD" "cat /srv/ml/eval_results_longctx_512k/$t/VERDICT 2>/dev/null || echo unknown")
    echo "  31B verdict: $verdict"
    case "$verdict" in
      pass)    echo "  → proceeding to v6-coder";;
      caveat)  echo "  → 31B passes hard gates but missed long-ctx goal; council recommended path may differ — HALTING for review"; exit 4;;
      *)       echo "  → 31B failed hard gate — HALTING cascade"; exit 5;;
    esac
  fi
done

echo
echo "================ CASCADE COMPLETE ================"
echo "  targets done: ${TARGETS[*]}"
echo "  see backup_models/docs/plans/gemma4_512k_plan_v2.md §'Decision matrix' for next steps"
