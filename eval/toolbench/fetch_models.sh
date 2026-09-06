#!/usr/bin/env bash
# Fetch the external GGUFs for the tool-eval-bench cohort, byte-size verified.
#
# Only files whose exact size was verified on 2026-09-05 are listed. A download
# is promoted into place ONLY on an exact byte match -- a short/resumed file is
# left in staging and reported, never silently used.
#
# The four in-house models (a3b-coder, a3b-coderx, omnimerge-v4, omnimerge-v6)
# are built by this repo's own pipelines and are not fetched here.
set -uo pipefail
M="${OMK_TB_MODELS:?OMK_TB_MODELS must point at the GGUF directory}"
export HF_HUB_DISABLE_XET=1        # xet has stalled mid-file on this host
export HF_HUB_ENABLE_HF_TRANSFER=1

# repo|upstream_file|expected_bytes|local_name
#
# NOTE the Qwen3.6-27B row. unsloth publishes that model in TWO repos under the
# IDENTICAL upstream filename `Qwen3.6-27B-Q4_K_M.gguf`:
#     unsloth/Qwen3.6-27B-GGUF      16817244384  851 tensors, NO NextN head
#     unsloth/Qwen3.6-27B-MTP-GGUF  17106773120  866 tensors, head present
# We fetch the MTP build and store it under a DISTINCT local name, so the two can
# never collide on disk and a later run cannot silently pick the head-less one.
# Getting this wrong costs more than speed: a head-less dense 27B is slow enough to
# trip tool-eval-bench's request timeout, which DROPS scenarios from the denominator
# and silently changes the metric. See README "Traps this tooling guards".
JOBS=(
 "unsloth/Qwen3.8-27B-GGUF|Qwen3.8-27B-UD-Q4_K_M.gguf|16464440224|Qwen3.8-27B-UD-Q4_K_M.gguf"
 "unsloth/Qwen3.6-27B-MTP-GGUF|Qwen3.6-27B-Q4_K_M.gguf|17106773120|Qwen3.6-27B-MTP-Q4_K_M.gguf"
 "bartowski/Ornith-1.5-35B-A3B-GGUF|Ornith-1.5-35B-A3B-IQ4_XS.gguf|19278554784|Ornith-1.5-35B-A3B-IQ4_XS.gguf"
 "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF|Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf|19699553920|Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf"
)
rc=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r repo file exp dest <<<"$j"
  dest="${dest:-$file}"
  if [ -f "$M/$dest" ]; then
    sz=$(stat -c%s "$M/$dest")
    [ "$sz" -eq "$exp" ] && { echo "OK        $dest (already present)"; continue; }
    echo "** WRONG SIZE $dest got=$sz want=$exp -- remove it and re-run"; rc=1; continue
  fi
  for i in 1 2 3 4 5 6; do
    echo "=== $file attempt $i $(date -Is) ==="
    hf download "$repo" "$file" --local-dir "$M/.dl_$dest" && break
    sleep 20
  done
  src="$M/.dl_$dest/$file"
  if [ -f "$src" ]; then
    sz=$(stat -c%s "$src")
    if [ "$sz" -eq "$exp" ]; then
      mv -f "$src" "$M/$dest"; rm -rf "$M/.dl_$dest"; echo "OK        $dest"
    else
      echo "** SIZE_MISMATCH $dest got=$sz want=$exp -- NOT promoted"; rc=1
    fi
  else
    echo "** DOWNLOAD_FAILED $dest"; rc=1
  fi
done
exit $rc
