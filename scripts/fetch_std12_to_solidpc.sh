#!/usr/bin/env bash
# fetch_std12_to_solidpc.sh — pull the STD12-v5 Q6_K + Q4_K_M from bs2 into the solidpc
# serving dir, under the deployment names (same convention the user set for std27).
#
# Staged as a dot-file and sha-verified BEFORE the rename, so a truncated transfer can
# never present itself as a servable model under the real name.
set -uo pipefail
ts(){ date '+%T %Z'; }
SRC=/mnt/sdc/ml/std12v5/STD12-v5-GGUF
DST=/shared/dev/opencoti/.opencoti/models
declare -A MAP=(
  [STD12-v5-bf16-Q6_K.gguf]=gemma-4-A4B-98e-v9-agentic-std12-it-Q6_K.gguf
  [STD12-v5-bf16-Q4_K_M.gguf]=gemma-4-A4B-98e-v9-agentic-std12-it-Q4_K_M.gguf
)
RC=0
for S in "${!MAP[@]}"; do
  D="${MAP[$S]}"
  WANT=$(ssh -o BatchMode=yes bs2 "sha256sum '$SRC/$S' | cut -d' ' -f1") || { echo "FATAL sha fetch $S"; RC=1; continue; }
  echo "[$(ts)] $S -> $D  (expect ${WANT:0:16})"
  if [ -f "$DST/$D" ] && [ "$(sha256sum "$DST/$D" | cut -d' ' -f1)" = "$WANT" ]; then
    echo "[$(ts)]   already present and matching, skipping"; continue
  fi
  scp -q "bs2:$SRC/$S" "$DST/.$D" || { echo "FATAL scp $S"; rm -f "$DST/.$D"; RC=1; continue; }
  GOT=$(sha256sum "$DST/.$D" | cut -d' ' -f1)
  if [ "$GOT" != "$WANT" ]; then
    echo "[$(ts)]   FATAL sha mismatch: got ${GOT:0:16} want ${WANT:0:16} — discarding"
    rm -f "$DST/.$D"; RC=1; continue
  fi
  mv "$DST/.$D" "$DST/$D"
  echo "$WANT  $D" > "$DST/$D.sha256"
  echo "[$(ts)]   OK $(stat -c%s "$DST/$D") bytes sha=${WANT:0:16}"
done
echo "[$(ts)] === STD12_FETCH_DONE rc=$RC ==="
exit $RC
