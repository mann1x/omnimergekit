#!/usr/bin/env bash
# exit 0 iff the LIVE (last) run block of the loop-cmp log has DONE or FATAL'd.
# Isolates the block after the final run header so a stale FATAL from a prior
# superseded run does not trip the watcher.
L=/mnt/sdc/ml/coderx_loopcmp_3tier.log
awk '/coderx loop-cmp 3-tier [0-9]/{buf=""} {buf=buf $0 ORS} END{printf "%s",buf}' "$L" \
  | grep -qE 'CODERX_LOOPCMP_DONE|^FATAL '
