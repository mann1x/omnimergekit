#!/bin/bash
# Wait for shared sweep to complete, then auto-launch PES sweep.
set -uo pipefail
SHARED_LOG=/srv/ml/logs/t172/sweep_shared_20260529_175543.log
TS=$(date +%Y%m%d_%H%M%S)
CHAIN_LOG=/srv/ml/logs/t172/chain_pes_${TS}.log
exec > "$CHAIN_LOG" 2>&1

echo "[$(date -Iseconds)] chain waiting for shared sweep completion..."
echo "  shared log: $SHARED_LOG"

until grep -qE "Phase 1 coarse sweep DONE for knob=shared|FATAL:" "$SHARED_LOG" 2>/dev/null; do
    sleep 30
done

if grep -q "FATAL:" "$SHARED_LOG"; then
    echo "[$(date -Iseconds)] shared sweep FAILED — NOT launching PES"
    exit 2
fi

echo "[$(date -Iseconds)] shared sweep DONE — launching PES sweep"
bash /srv/ml/scripts/sweep_alpha_t172.sh --knob pes --alphas 1.10,1.20,1.30 &
PID=$!
echo "[$(date -Iseconds)] PES sweep PID=$PID"
wait $PID
echo "[$(date -Iseconds)] PES sweep exit=$?"
