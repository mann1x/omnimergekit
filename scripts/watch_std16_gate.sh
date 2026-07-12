#!/usr/bin/env bash
# watch_std16_gate.sh — track the STD16 loop-gate sweep until its driver exits. Emits the
# running fails/48 table and flags any looper (fails>0 -> balanced-imatrix rebuild). Background.
WORK=/mnt/sdc/ml/std16_gate
DRV=1068954
ts(){ date '+%T %Z'; }
prev=-1
while kill -0 "$DRV" 2>/dev/null; do
  n=$(ls "$WORK"/done/*.done 2>/dev/null | wc -l)
  if [ "$n" != "$prev" ]; then
    echo "[gate-watch $(ts)] $n/18 tiers gated"
    [ -f "$WORK/SUMMARY.tsv" ] && sort "$WORK/SUMMARY.tsv" | sed 's/^/    /'
    prev=$n
  fi
  sleep 120
done
echo "[gate-watch $(ts)] ===== STD16 GATE DRIVER EXITED ====="
sort "$WORK/SUMMARY.tsv" 2>/dev/null
echo "=== LOOPERS (fails>0 -> balanced-imatrix rebuild) ==="
grep -vE "fails=0/48[[:space:]]+fails=0/48" "$WORK/SUMMARY.tsv" 2>/dev/null | grep -vE "^$" || echo "  none — all tiers 0/48 clean"
echo "STD16_GATE_WATCH_DONE"
