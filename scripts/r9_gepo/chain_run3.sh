#!/usr/bin/env bash
# Wait for the LCB repeat draw to release GPU1, then launch GEPO run3.
#
# Waits on the OBSERVED condition (process gone AND both GPUs actually free),
# not on a sleep -- a sleep is not a readiness predicate. The LCB draw's own
# outcome does not gate run3: they are independent experiments sharing a GPU.
#
# pgrep patterns are bracketed ([l]cb_repeat) so the pattern cannot match this
# script's own command line -- the bs2 self-match trap that returns 255.
set -uo pipefail

W=/mnt/sdc/ml/brevity/gepo
LOG=$W/chain_run3.log
DEADLINE=$(( $(date +%s) + 4*3600 ))

say(){ echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG"; }

gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -dc '0-9'; }

say "=== chain_run3: waiting for GPU1 to free ==="
while :; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    say "ABORT: 4h deadline passed, LCB draw still holding GPU1"; exit 1
  fi
  running=$(pgrep -f "[l]cb_repeat.sh" | wc -l)
  g1=$(gpu_used 1); g0=$(gpu_used 0)
  if [ "$running" -eq 0 ] && [ "${g1:-99999}" -lt 2000 ]; then
    say "GPU1 released (proc=$running g1=${g1}MiB g0=${g0}MiB)"
    break
  fi
  sleep 60
done

# run3 needs BOTH GPUs (DDP, CUDA_VISIBLE_DEVICES=0,1), explicitly authorised.
g0=$(gpu_used 0); g1=$(gpu_used 1)
if [ "${g0:-99999}" -gt 2000 ]; then
  say "REFUSE: GPU0 busy (${g0}MiB) -- run3 needs both GPUs. Not launching."; exit 2
fi
say "both GPUs free (g0=${g0}MiB g1=${g1}MiB)"

# Never train over an existing adapter; the launcher also refuses, this is belt.
if [ -e "$W/run3" ]; then
  say "REFUSE: $W/run3 already exists -- adapters are never overwritten."; exit 3
fi

say "LAUNCH r9_gepo_run3.sh (router in scope, smoke-gated)"
bash /srv/ml/repos/omnimergekit/scripts/r9_gepo_run3.sh >>"$LOG" 2>&1
rc=$?
say "r9_gepo_run3.sh rc=$rc"
say "CHAIN_RUN3_EXIT $rc"
