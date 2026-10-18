#!/usr/bin/env bash
set -uo pipefail
cd /srv/ml/repos/omnimergekit
echo ">>> $(date -u +%H:%M:%S) JackDeltaCoder-9B Q6_K build start"
df -BG /mnt/sdc | tail -1
/root/anaconda3/envs/omnimergekit/bin/python scripts/quantize_gguf.py \
  --model /mnt/sdc/oxopus/JackDeltaCoder-9B \
  --output-dir /mnt/sdc/oxopus/gguf \
  --only Q6_K \
  --imatrix-chunks 128 \
  --base-model-id Jackrong/Qwopus3.5-9B-Coder \
  --no-upload \
  --keep-local \
  --ngl 99
rc=$?
echo ">>> $(date -u +%H:%M:%S) quantize_gguf rc=$rc"
echo "=== artefacts ==="
ls -la /mnt/sdc/oxopus/gguf/ 2>/dev/null | grep -E "gguf|imatrix"
[ $rc -eq 0 ] && echo "JACKDELTA_Q6K_DONE" || echo "JACKDELTA_Q6K_FAILED"
