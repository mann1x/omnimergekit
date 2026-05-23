#!/usr/bin/env bash
# stack_canary_swap_and_run.sh — install a candidate vLLM, run 4-doc canary,
# restore stack@2. Safe for iterative stack debugging (T91).
#
# Usage:
#   bash scripts/stack_canary_swap_and_run.sh <label> <pip_install_spec> [--no-restore]
#
# Examples:
#   # H1a — stock vLLM 0.20.2 from PyPI
#   bash scripts/stack_canary_swap_and_run.sh stack3a_stock_020_2 'vllm==0.20.2'
#
#   # H1b — local wheel with cherry-pick #42250
#   bash scripts/stack_canary_swap_and_run.sh stack3b_42250_only /path/to/wheel.whl
#
# Default: always restores stack@2 wheel at the end. Use --no-restore to
# leave the candidate vLLM installed for follow-up runs.

set -uo pipefail

ROOT="/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models"
OMK="/shared/dev/omnimergekit"
STACK2_WHEEL="$ROOT/wheels/gemma4-moe-stack-v2/vllm-0.21.1rc1.dev178+g3d92852eb.cu132.sm86-cp311-cp311-linux_x86_64.whl"
PIP="/root/anaconda3/envs/vllm/bin/pip"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <label> <pip_install_spec> [--no-restore]"
    echo "Example: $0 stack3a_stock_020_2 'vllm==0.20.2'"
    exit 2
fi

LABEL="$1"
SPEC="$2"
RESTORE="yes"
if [ "${3:-}" = "--no-restore" ]; then
    RESTORE="no"
fi

LOGTS=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT/logs/stack_canary_swap_${LABEL}_${LOGTS}.log"
mkdir -p "$ROOT/logs"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== H? Stack swap + canary run — label: $LABEL ==="
log "spec: $SPEC"
log "restore stack@2: $RESTORE"
log ""
log "=== Pre-swap vLLM state ==="
$PIP show vllm 2>/dev/null | grep -E "Name|Version|Location" | tee -a "$LOG"
log ""

# Confirm restore wheel is present BEFORE swap
if [ ! -f "$STACK2_WHEEL" ]; then
    log "ABORT: stack@2 restore wheel missing at $STACK2_WHEEL"
    exit 3
fi
log "Stack@2 restore wheel verified: $(ls -lh "$STACK2_WHEEL" | awk '{print $5}')"

# Pre-flight GPU check
GPU_USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${GPU_USED_MB:-0}" -gt 2000 ]; then
    log "ABORT: GPU has ${GPU_USED_MB} MiB in use; free it first"
    exit 3
fi

log ""
log "=== Step 1/4: Install candidate vLLM ($SPEC) ==="
$PIP install --force-reinstall --no-deps "$SPEC" 2>&1 | tee -a "$LOG" | tail -30
INSTALL_RC=${PIPESTATUS[0]}
if [ "$INSTALL_RC" -ne 0 ]; then
    log "ABORT: pip install failed (rc=$INSTALL_RC)"
    if [ "$RESTORE" = "yes" ]; then
        log "Attempting restore of stack@2..."
        $PIP install --force-reinstall --no-deps "$STACK2_WHEEL" 2>&1 | tee -a "$LOG" | tail -10
    fi
    exit 3
fi

log ""
log "=== Step 2/4: Verify install ==="
$PIP show vllm 2>/dev/null | grep -E "Name|Version" | tee -a "$LOG"
NEW_VER=$($PIP show vllm 2>/dev/null | awk -F': ' '/^Version/ {print $2}')
log "Installed: vllm $NEW_VER"

log ""
log "=== Step 3/4: Run 4-doc canary ==="
bash "$OMK/scripts/stack_canary_4doc_run.sh" "$LABEL" 2>&1 | tee -a "$LOG"
CANARY_RC=${PIPESTATUS[0]}
log "canary rc=$CANARY_RC"

log ""
log "=== Step 4/4: Restore stack@2 ==="
if [ "$RESTORE" = "yes" ]; then
    $PIP install --force-reinstall --no-deps "$STACK2_WHEEL" 2>&1 | tee -a "$LOG" | tail -10
    RESTORE_RC=${PIPESTATUS[0]}
    if [ "$RESTORE_RC" -ne 0 ]; then
        log "WARN: stack@2 restore failed (rc=$RESTORE_RC) — manual recovery needed"
    else
        log "stack@2 restored OK"
        $PIP show vllm 2>/dev/null | grep -E "Name|Version" | tee -a "$LOG"
    fi
else
    log "RESTORE=no — candidate vLLM left in place ($NEW_VER)"
fi

log ""
case $CANARY_RC in
    0) log "FINAL VERDICT for $LABEL: ALL_PASS (4/4)" ;;
    2) log "FINAL VERDICT for $LABEL: ANY_FAIL — see canary_result.json" ;;
    3) log "FINAL VERDICT for $LABEL: SETUP_ERROR" ;;
    *) log "FINAL VERDICT for $LABEL: unexpected rc=$CANARY_RC" ;;
esac

exit $CANARY_RC
