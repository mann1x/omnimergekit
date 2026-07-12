#!/bin/bash
# Restore HE+ and MPE _p8_OLD result dirs back to live names.
# These are the original parallel=8 CLEAN results — valid because
# those benches don't trip the per-slot-ctx < thinking-budget bug.
# Only IFEval needs the re-run.
set -uo pipefail
RES=/srv/ml/eval_results_tracks_2_3

CELLS=(
  a2eac_calibonly-62e-fc15_25-p8-s1_0p1_20
  a2eac_9bench-62e-fc15_25-p8-s1_0p1_20
  a2eac_ifheavy-62e-fc15_25-p8-s1_0p1_20
  a2kdonly_calibonly-62e-fc15_25-p8-s1_0p1_20
  a2kdonly_9bench-62e-fc15_25-p8-s1_0p1_20
  a2kdonly_ifheavy-62e-fc15_25-p8-s1_0p1_20
  a2rkd_calibonly-62e-fc15_25-p8-s1_0p1_20
  a2rkd_9bench-62e-fc15_25-p8-s1_0p1_20
  a2rkd_ifheavy-62e-fc15_25-p8-s1_0p1_20
  gemma-4-A4B-62e-fc15_25-p8-pristine-it
  gemma-4-A4B-62e-fc15_25-p8-shared110-it
  gemma-4-A4B-62e-fc15_25-p8-shared120-it
  gemma-4-A4B-62e-fc15_25-p8-shared130-it
  gemma-4-A4B-62e-fc15_25-p8-pes110-it
  gemma-4-A4B-62e-fc15_25-p8-pes120-it
  gemma-4-A4B-62e-fc15_25-p8-pes130-it
  pes1_10-62e-fc15_25-p8
)
KEEP_BENCHES=(humanevalplus_full multipl_e_100)
for cell in "${CELLS[@]}"; do
  for tpl in "${KEEP_BENCHES[@]}"; do
    LIVE="$RES/$tpl/$cell"
    OLD="$RES/$tpl/${cell}_p8_OLD"
    if [ -d "$OLD" ]; then
      # If a partial new dir exists from the killed sweep, remove it
      if [ -d "$LIVE" ]; then rm -rf "$LIVE"; fi
      mv "$OLD" "$LIVE"
      echo "restored $tpl/$cell  (parallel=8 result kept)"
    elif [ ! -d "$LIVE" ]; then
      echo "MISSING $tpl/$cell  (no LIVE, no _p8_OLD)"
    fi
  done
done
echo
echo "=== sanity: count summary.json in each kept bench ==="
for tpl in "${KEEP_BENCHES[@]}"; do
  n=$(find "$RES/$tpl" -maxdepth 2 -name summary.json 2>/dev/null | wc -l)
  echo "  $tpl  summaries=$n"
done
