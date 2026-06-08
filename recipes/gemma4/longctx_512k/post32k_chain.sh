#!/usr/bin/env bash
# post32k_chain.sh — runs ON bs2, detached. Waits for the flagship DDP@32k 250M
# run to exit, then runs the gated PP/64k loss-equivalence+throughput smoke.
# Durable (nohup/setsid) so it survives the controlling Claude session; the
# Monitor on the solidpc side only TAILS this log to notify — the work happens here.
#
# Stops at the PP smoke. RULER + the 500M continuation are deliberately NOT
# auto-chained: RULER validates the 32k YaRN extension (gate before investing the
# next 250M) and the 500M resume is a 15h commitment — both are reviewed first.
set -uo pipefail
LOG=/srv/ml/longctx/post32k.log
exec >>"$LOG" 2>&1
RUNLOG=/srv/ml/longctx/ddp32k_fa2.log
PAT="ckpt_98e_ddp32k_fa2"

echo "=== post32k_chain armed $(date '+%F %T %Z') — waiting for flagship run to exit ==="
# Wait until no trainer process referencing the flagship ckpt-dir remains.
while pgrep -f "$PAT" >/dev/null 2>&1; do sleep 60; done
echo "=== FLAGSHIP EXITED $(date '+%F %T %Z') ==="
echo "--- last 3 flagship log lines ---"
tail -n 3 "$RUNLOG" 2>/dev/null

if grep -q "training complete" "$RUNLOG" 2>/dev/null; then
  echo "=== run COMPLETED cleanly — running PP/64k smoke (GPUs free) ==="
  bash /srv/ml/scripts/pp_smoke_driver.sh
  echo "=== PP SMOKE DONE $(date '+%F %T %Z') ==="
else
  echo "=== WARNING: flagship exited WITHOUT 'training complete' — crash/stop suspected."
  echo "=== NOT auto-running the PP smoke. Inspect $RUNLOG before proceeding. ==="
fi
echo "=== post32k_chain end $(date '+%F %T %Z') ==="
