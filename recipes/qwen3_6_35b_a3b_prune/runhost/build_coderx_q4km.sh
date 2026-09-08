#!/usr/bin/env bash
# First CoderX tier through the CANONICAL pipeline. Two jobs:
#   1) it is a shipped tier (Q4_K_M = the :latest tier), and
#   2) it is the missing cell for the draft_num_predict decision -- CoderX on a CONSUMER
#      GPU. bs2's Blackwell peaked at n=3 (n=8 = -23%); the 3090 peaked at n=8 (+71%) but
#      on a DIFFERENT model, so model-vs-GPU is still confounded. 16 GB copies to solidpc.
# Built FROM the publish dir so filenames are release-correct (bf16-source-dir == target repo).
# No --force-imatrix: IMATRIX_EXCLUDE already encodes the Q4/Q5/Q6 noimat crossover (#566).
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit
SRC=/mnt/sdc/ream-work/publish/Qwen3.6-27B-A3B-CoderX
OUT=/mnt/sdc/ream-work/gguf_coderx
mkdir -p "$OUT"
free=$(df -BG --output=avail /mnt/sdc | tail -1 | tr -dc 0-9)
[ "${free:-0}" -ge 120 ] || { echo "REFUSE: /mnt/sdc ${free}G free, need >=120G for F16+Q4_K_M"; exit 1; }
echo "[build $(date -u +%H:%M:%SZ)] starting; /mnt/sdc ${free}G free"
"$PY" "$OMK/scripts/quantize_gguf.py" \
    --model "$SRC" \
    --output-dir "$OUT" \
    --only Q4_K_M \
    --no-upload --keep-local \
    --base-model-id Qwen/Qwen3.6-35B-A3B 2>&1 | tail -40
echo "[build $(date -u +%H:%M:%SZ)] rc=$? ; contents:"
ls -la "$OUT"
echo "CODERX_Q4KM_DONE"
