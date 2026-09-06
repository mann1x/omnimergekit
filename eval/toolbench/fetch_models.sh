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

# repo|file|expected_bytes
JOBS=(
 "unsloth/Qwen3.8-27B-GGUF|Qwen3.8-27B-UD-Q4_K_M.gguf|16464440224"
 "unsloth/Qwen3.6-27B-GGUF|Qwen3.6-27B-Q4_K_M.gguf|16817244384"
 "bartowski/Ornith-1.5-35B-A3B-GGUF|Ornith-1.5-35B-A3B-IQ4_XS.gguf|19278554784"
 "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF|Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf|19699553920"
)
rc=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r repo file exp <<<"$j"
  if [ -f "$M/$file" ]; then
    sz=$(stat -c%s "$M/$file")
    [ "$sz" -eq "$exp" ] && { echo "OK        $file (already present)"; continue; }
    echo "** WRONG SIZE $file got=$sz want=$exp -- remove it and re-run"; rc=1; continue
  fi
  for i in 1 2 3 4 5 6; do
    echo "=== $file attempt $i $(date -Is) ==="
    hf download "$repo" "$file" --local-dir "$M/.dl_$file" && break
    sleep 20
  done
  src="$M/.dl_$file/$file"
  if [ -f "$src" ]; then
    sz=$(stat -c%s "$src")
    if [ "$sz" -eq "$exp" ]; then
      mv -f "$src" "$M/$file"; rm -rf "$M/.dl_$file"; echo "OK        $file"
    else
      echo "** SIZE_MISMATCH $file got=$sz want=$exp -- NOT promoted"; rc=1
    fi
  else
    echo "** DOWNLOAD_FAILED $file"; rc=1
  fi
done
exit $rc
