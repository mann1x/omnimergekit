#!/bin/bash
# auto_chain_watcher.sh — wait for MTP build (PID arg), then fire eval chain.
#
# Usage:  bash auto_chain_watcher.sh <BUILD_PID>
#         (intended for: nohup bash auto_chain_watcher.sh 948 > /workspace/logs/watcher.log 2>&1 & disown)
#
# Polls every 60s for the build PID to die. When it does:
#   - confirms /workspace/out/Qwen3.6-27B-Omnimerge-v4-Q6_K.gguf exists (build succeeded for at least Q6_K)
#   - logs final MTP build outputs (tier list)
#   - launches pod_v4_q6k_eval_chain.sh in nohup
#   - writes /workspace/eval_chain.pid for downstream monitors
# If the Q6_K isn't there → logs the failure and exits without firing.

set -u
BUILD_PID="${1:?usage: $0 BUILD_PID}"
WORK=/workspace
LOG=$WORK/logs/auto_chain_watcher.log
mkdir -p "$WORK/logs"

echo "[$(date +%H:%M:%S)] auto_chain_watcher armed — waiting on build PID $BUILD_PID" >> "$LOG"

# Wait loop — poll every 60s. Cap total wait at 6 hours (any longer = build hung).
DEADLINE=$(($(date +%s) + 6 * 3600))
while kill -0 "$BUILD_PID" 2>/dev/null; do
    if [ $(date +%s) -gt $DEADLINE ]; then
        echo "[$(date +%H:%M:%S)] WATCHER: 6h deadline elapsed while build still alive; aborting watcher (build keeps running)" >> "$LOG"
        exit 1
    fi
    sleep 60
done
echo "[$(date +%H:%M:%S)] build PID $BUILD_PID exited" >> "$LOG"

# Brief pause for the build's final filesystem writes
sleep 5

# Verify MTP Q6_K exists
MTP_Q6K=$WORK/out/Qwen3.6-27B-Omnimerge-v4-Q6_K.gguf
if [ ! -f "$MTP_Q6K" ]; then
    echo "[$(date +%H:%M:%S)] FAIL: $MTP_Q6K missing — build did not produce Q6_K. Eval skipped." >> "$LOG"
    echo "[$(date +%H:%M:%S)] /workspace/out contents:" >> "$LOG"
    ls -la $WORK/out/ >> "$LOG" 2>&1
    exit 1
fi
echo "[$(date +%H:%M:%S)] MTP Q6_K verified: $(du -h $MTP_Q6K | cut -f1)" >> "$LOG"
echo "[$(date +%H:%M:%S)] All MTP outputs:" >> "$LOG"
ls -la $WORK/out/ >> "$LOG" 2>&1

# Inherit HF_TOKEN from the MTP launcher's saved env
HF_TOKEN_VAL=$(grep -m1 '^export HF_TOKEN=' $WORK/mtp_launch.sh 2>/dev/null | cut -d'"' -f2)
if [ -z "$HF_TOKEN_VAL" ]; then
    echo "[$(date +%H:%M:%S)] WARNING: HF_TOKEN not recoverable from mtp_launch.sh; using shell env" >> "$LOG"
    HF_TOKEN_VAL="${HF_TOKEN:-}"
fi

# Fire the eval chain in nohup
EVAL_LOG=$WORK/logs/eval_chain_$(date +%Y%m%d_%H%M%S).log
echo "[$(date +%H:%M:%S)] firing pod_v4_q6k_eval_chain.sh → $EVAL_LOG" >> "$LOG"
HF_TOKEN="$HF_TOKEN_VAL" HF_HUB_ENABLE_HF_TRANSFER=1 \
    nohup bash $WORK/scripts/pod_v4_q6k_eval_chain.sh > "$EVAL_LOG" 2>&1 &
CHAIN_PID=$!
disown $CHAIN_PID
echo "$CHAIN_PID" > $WORK/eval_chain.pid
echo "[$(date +%H:%M:%S)] eval_chain.pid = $CHAIN_PID  log = $EVAL_LOG" >> "$LOG"
echo "[$(date +%H:%M:%S)] auto_chain_watcher done" >> "$LOG"
