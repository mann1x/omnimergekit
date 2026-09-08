#!/usr/bin/env bash
# Reap F16 intermediates once their Q6_K is finished, so the 5-arm chain does not
# strand itself on disk.
#
# quant_arms_imat.sh assumed quantize_gguf drops its own F16 ("no --keep-local"). It does
# not: armD left a 52 G F16 next to its 21 G Q6_K. Five arms would need ~365 G and only
# ~113 G was ever free, so the chain's own preflight (avail < 85 G => ABORT) would have
# killed it at armB. The chain is RUNNING and must not be edited, so this reaps alongside.
#
# The delete gate is the .sha256 SIDECAR, not the Q6_K file. The Q6_K exists from the
# moment llama-quantize starts writing it; the sidecar is written after quantize_gguf has
# finished with the F16. Deleting on the Q6_K alone would pull the source out from under
# a live quantize. F16 is a pure intermediate: reproducible from the arm weights in ~3 min.
set -uo pipefail
GG=/mnt/sdc/ream-work/gguf
LOG=/mnt/sdc/ream-work/reap_f16.log
exec >>"$LOG" 2>&1
echo "=== reaper start $(date -u +%F' '%T) ==="

for _ in $(seq 1 720); do          # 720 x 60s = 12 h, well past the 5-arm chain
  for d in "$GG"/*_imat; do
    [ -d "$d" ] || continue
    q=$(ls "$d"/*-Q6_K.gguf 2>/dev/null | head -1)
    f=$(ls "$d"/*-F16.gguf 2>/dev/null | head -1)
    [ -n "$q" ] && [ -n "$f" ] || continue
    [ -s "${q}.sha256" ] || continue          # quantize still holds the F16
    [ "$(head -c4 "$q")" = "GGUF" ] || { echo "  REFUSE $(basename "$d"): Q6_K not a GGUF"; continue; }
    echo "  [$(date -u +%H:%M:%S)] reaping $(basename "$f") ($(du -h "$f" | cut -f1)); \
Q6_K $(du -h "$q" | cut -f1) + sha256 present"
    rm -f "$f"
    echo "    /mnt/sdc now $(df -BG --output=avail /mnt/sdc | tail -1 | tr -dc 0-9)G free"
  done
  grep -q ">>> QUANT_ARMS_DONE" /mnt/sdc/ream-work/quant_arms_imat.log 2>/dev/null && {
    echo "=== chain done; final sweep then exit ==="; }
  sleep 60
done
echo ">>> REAPER_EXIT $(date -u +%F' '%T)"
