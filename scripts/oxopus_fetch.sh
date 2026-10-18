#!/bin/bash
# OxOpusCoder-9B: fetch the three models with aria2c (segmented, resumable).
#   base      Jackrong/Qwopus3.5-9B-Coder   775 tensors, MTP intact, complete tokenizer
#   source    OrionLLM/OxCoder-9B           760 tensors, no MTP
#   task-base Qwen/Qwen3.5-9B               common ancestor -- delta computed vs THIS
#
# aria2c per user preference: multi-connection segmented with per-segment resume, so a
# 19 GB shard set lands in a fraction of single-stream time and a dropped connection
# resumes the segment instead of restarting.
#
# All three repos are PUBLIC -- no Authorization header, so no token in the process table.
#
# Gate on SIZE and on the safetensors COUNT, never on the exit code: an earlier
# `hf download` run exited 0 having fetched ~300 KB because bare words after the repo
# are parsed as filenames-to-fetch, not exclude patterns.
set -uo pipefail
W=/mnt/sdc/oxopus
MIN_GB=15
rm -rf "$W/models"; mkdir -p "$W/models"

# build the aria2c input file from the manifest (repo \t dir \t file \t size)
IN="$W/aria.in"; : > "$IN"
while IFS=$'\t' read -r repo dir file size; do
  [ -n "${repo:-}" ] || continue
  mkdir -p "$W/models/$dir/$(dirname "$file")"
  printf "https://huggingface.co/%s/resolve/main/%s\n  dir=%s\n  out=%s\n" \
         "$repo" "$file" "$W/models/$dir/$(dirname "$file")" "$(basename "$file")" >> "$IN"
done < "$W/files.tsv"
echo ">>> $(date -u +%H:%M:%S) aria2c starting: $(grep -c "^https" "$IN") files"

aria2c -i "$IN" \
  -x 16 -s 16 -k 1M --continue=true \
  --max-concurrent-downloads=3 \
  --auto-file-renaming=false --allow-overwrite=true \
  --retry-wait=5 --max-tries=10 --timeout=60 \
  --summary-interval=30 --console-log-level=warn
rc=$?
echo ">>> $(date -u +%H:%M:%S) aria2c rc=$rc"

fail=0
for dir in Qwopus3.5-9B-Coder OxCoder-9B Qwen3.5-9B; do
  d="$W/models/$dir"
  gb=$(du -sBG "$d" 2>/dev/null | cut -f1 | tr -dc "0-9"); gb=${gb:-0}
  n=$(ls "$d"/*.safetensors 2>/dev/null | wc -l)
  echo ">>> $dir: ${gb}G, $n safetensors"
  if [ "$gb" -lt "$MIN_GB" ] || [ "$n" -lt 1 ]; then
    echo "FETCH_FAIL $dir -- ${gb}G / $n safetensors, expected >=${MIN_GB}G and >=1"; fail=1
  fi
done
[ "$fail" -eq 0 ] || { echo "OXOPUS_FETCH_FAILED"; exit 2; }
du -sh "$W"/models/*
echo "OXOPUS_FETCH_DONE"
