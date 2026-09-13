#!/usr/bin/env bash
# Wait for the BFCL composite cohort, then print its summary.
# NOTE: `a=$(grep -c X f || echo 0)` prints "0\n0" when the count is zero
# (grep -c prints 0 AND exits 1). Use grep -c alone; it always prints a number.
set -uo pipefail
L=/mnt/sdc/harness/logs
for i in $(seq 1 720); do
  a=$(grep -ac "BFCL_armA_comp_DONE" "$L/run_armA_comp.log" 2>/dev/null); a=${a:-0}
  b=$(grep -ac "BFCL_armB_comp_DONE" "$L/run_armB_comp.log" 2>/dev/null); b=${b:-0}
  if [ "$a" -ge 1 ] && [ "$b" -ge 1 ]; then echo "COMPOSITE_BOTH_DONE $(date -u +%FT%TZ)"; exit 0; fi
  sleep 30
done
echo "COMPOSITE_WAIT_TIMEOUT $(date -u +%FT%TZ)"; exit 1
