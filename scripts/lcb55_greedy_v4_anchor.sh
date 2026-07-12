#!/usr/bin/env bash
# lcb55_greedy_v4_anchor.sh — clean greedy LCB-55 anchor for 128e Q6_K on b9700.
# The Part C greedy LCB cell used the PLAIN lcb_medium_55 template (no
# --reasoning-budget) and truncated 9/55 (finish=length) -> artificially low
# 47/55 = 85.45%. This re-runs the SAME bench on lcb_medium_55_v4 (which forces
# --reasoning-budget 12288) to get an untruncated anchor. Greedy (no --sampler).
# Pinned to the IDLE GPU0; vendor recovery owns GPU1 -> never co-locate (bug-468).
# Result lands as a sibling lcb_medium_55_v4/ subdir under the greedy variant dir,
# preserving the original (truncated) lcb_medium_55/ cell for the record.
set -uo pipefail
export OMK_ROOT=/srv/ml/repos/omnimergekit-canonical
export OMK_WS=/srv/ml/partc_b9700
export OMK_TOKENIZER=/srv/ml/google/gemma-4-26B-A4B-it
export LLAMA_BIN=/mnt/sdc/ml/llama.cpp-b9700/build/bin
export OMK_GPUS=0
export OMK_GPU_WAIT_S=300
cd "$OMK_ROOT/eval" || { echo "FATAL cannot cd $OMK_ROOT/eval"; exit 7; }
echo "[lcb-v4-anchor $(date '+%T %Z')] launching --only lcb_medium_55_v4, GREEDY, OMK_GPUS=0 port 8091"
./eval_suite_llama.sh \
  --variant 128e_b9700_greedy \
  --gguf /mnt/sdc/ml/eval_gguf/128e-Q6_K.gguf \
  --port 8091 \
  --only lcb_medium_55_v4
echo "[lcb-v4-anchor $(date '+%T %Z')] LCB_V4_ANCHOR_DONE rc=$?"
