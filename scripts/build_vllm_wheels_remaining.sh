#!/bin/bash
# build_vllm_wheels_remaining.sh — kick off the non-sm86 arches once the
# sm86 wheel is on disk. Separated so we can start sm86 alone, run the
# v2-stack canary against its wheel on GPU as soon as the live LCB run
# finishes, and let the remaining 4 arches grind on CPU in parallel.
#
# Sequence: sm89 → sm90 → sm100 → sm120 (host C++ ccache hits dominate
# after sm86 fills the cache; per-arch CUDA misses are unavoidable).
#
# Usage:
#   bash scripts/build_vllm_wheels_remaining.sh
#   MAX_JOBS=8 bash scripts/build_vllm_wheels_remaining.sh   # use more cores

set -euo pipefail

WHEELS_DIR=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/wheels/gemma4-moe-stack-v2
LOG_DIR="$WHEELS_DIR/build_logs"

# Pre-flight: refuse to start if sm86 wheel isn't already on disk.
if ! ls "$WHEELS_DIR"/*sm86*.whl >/dev/null 2>&1; then
    echo "ERROR: no sm86 wheel in $WHEELS_DIR yet — finish sm86 first" >&2
    exit 1
fi

# Pre-flight: refuse to start if an sm86 build is still running.
if pgrep -f 'build_vllm_wheels.sh sm86' >/dev/null 2>&1; then
    echo "ERROR: sm86 build still running — wait for it to finish" >&2
    exit 2
fi

mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/remaining_${STAMP}.log"

cd /shared/dev/omnimergekit
echo "remaining arches → $LOG"
# Default MAX_JOBS=8 here (vs 6 for sm86) because the live LCB run will
# typically be done by the time this kicks off, so we have full CPU.
nohup env MAX_JOBS="${MAX_JOBS:-8}" bash scripts/build_vllm_wheels.sh \
    sm89 sm90 sm100 sm120 > "$LOG" 2>&1 &
disown
sleep 3
echo "launched (PIDs):"
pgrep -af 'build_vllm_wheels.sh sm89' | grep -v grep | head -3 || true
echo "log: $LOG"
