#!/usr/bin/env bash
# watch_fast_12b.sh — track the fast (32k-cap, 48k-budget) 12B test until its driver exits.
# Parses each cell's stdout log for per-seed [gemma_card seed=...] lines: completed count,
# loops, runaways, and max ans tokens (should now be <=~32k vs the no-cap 392k). Background.
WORK=/srv/ml/agentic_loop_12b_test/full_fast48k
DRV=1067714
ts(){ date '+%T %Z'; }
cellstat(){ # $1=cell -> "done=N loop=L run=R maxans=A"
  local lg="$WORK/$1.log"
  [ -f "$lg" ] || { echo "done=0"; return; }
  local d l r a
  d=$(grep -c "gemma_card seed=" "$lg" 2>/dev/null)
  l=$(grep -c "loop=True" "$lg" 2>/dev/null)
  r=$(grep -c "RUNAWAY" "$lg" 2>/dev/null)
  a=$(grep -oE "ans=[0-9]+" "$lg" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
  echo "done=$d loop=$l run=$r maxans=${a:-0}"
}
prev=""
while kill -0 "$DRV" 2>/dev/null; do
  cur=""
  for c in embedded reinject_off pr35 pr35_ptfalse; do cur="$cur | $c:$(cellstat "$c")"; done
  [ "$cur" != "$prev" ] && { echo "[watch $(ts)]$cur"; prev="$cur"; }
  sleep 90
done
echo "[watch $(ts)] ===== DRIVER $DRV EXITED ====="
tail -4 "$WORK/../run_fast.driver.log" 2>/dev/null
for c in embedded reinject_off pr35 pr35_ptfalse; do echo "  $c: $(cellstat "$c")"; done
echo "FAST_12B_WATCH_DONE"
