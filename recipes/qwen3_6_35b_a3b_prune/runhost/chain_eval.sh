#!/usr/bin/env bash
# Queue the R2 eval directly behind the quant chain. Last time the arms finished and GPU1 then
# sat idle for 5h because the eval was not queued behind the build -- this closes that gap.
# Waits on the QUANT_ARMS_DONE sentinel (not a timer, not an exit code), and only starts once
# the GPU is actually free, since the arm-F build shares GPU1 with this eval.
set -uo pipefail
WORK=/mnt/sdc/ream-work
LOG=$WORK/chain_eval.log
exec >>"$LOG" 2>&1
echo "=== chain_eval start $(date -u +%F' '%T) ==="

waited=0
until grep -q ">>> QUANT_ARMS_DONE" "$WORK/quant_arms_imat.log" 2>/dev/null; do
  sleep 60; waited=$((waited+60))
  [ $((waited % 900)) -eq 0 ] && echo "[$(date -u +%H:%M:%S)] waiting on quants (${waited}s)"
  [ $waited -ge 21600 ] && { echo "ABORT: no QUANT_ARMS_DONE after 6h"; exit 3; }
done
echo "quant sentinel seen after ${waited}s: $(grep -a QUANT_ARMS_DONE "$WORK/quant_arms_imat.log" | tail -1)"

# The quant chain can finish while the arm-F builder is still releasing the GPU.
for _ in $(seq 1 30); do
  free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free" -ge 60000 ] && break
  echo "GPU1 only ${free}MiB free, waiting"; sleep 60
done

bash "$WORK/eval_ream_arms.sh"
echo "=== chain_eval done $(date -u +%F' '%T) rc=$? ==="
