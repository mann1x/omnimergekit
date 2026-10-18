#!/usr/bin/env bash
set -euo pipefail
M=/mnt/sdc/oxopus/models
OUT=/mnt/sdc/oxopus/OxOpusCoder-9B
cd /srv/ml/repos/omnimergekit
echo ">>> $(date -u +%H:%M:%S) merge arm A start"
df -BG /mnt/sdc | tail -1
/root/anaconda3/envs/omnimergekit/bin/python omnimergekit.py \
  --base "$M/Qwopus3.5-9B-Coder" \
  --task-base "$M/Qwen3.5-9B" \
  --source "$M/OxCoder-9B" \
  --weights 0.40 \
  --method omnimerge_v2 \
  --density 0.53 \
  --darex-q 0.75 \
  --seed 42 \
  --no-auto-mlp-skip \
  --skip-patterns visual.,mtp. \
  --output "$OUT"
rc=$?
echo ">>> $(date -u +%H:%M:%S) omnimergekit rc=$rc"
[ $rc -ne 0 ] && { echo "OXOPUS_MERGE_FAILED"; exit $rc; }
du -sh "$OUT"
echo "OXOPUS_MERGE_DONE"
