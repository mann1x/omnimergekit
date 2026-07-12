#!/usr/bin/env bash
# Wait for the 2x2 to finish (frees GPU0), then run the balanced-imatrix proof.
set -u
echo "[chain] waiting for 2x2 to finish... $(date -u)"
while pgrep -f "cap_vs_artifact_2x2" >/dev/null 2>&1; do sleep 60; done
echo "[chain] 2x2 done; GPU0 free; launching balanced-imatrix proof $(date -u)"
bash /srv/ml/scripts/v7_publish/validate_balanced_imatrix.sh
echo "[chain] proof complete $(date -u)"
