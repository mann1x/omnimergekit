#!/usr/bin/env bash
# lcb55_vendor_v4_anchor.sh — clean vendor_base (recommended sampler, temp 1.0)
# LCB-55 anchor for 128e Q6_K on b9700. The Part C vendor recovery LCB cell used
# the PLAIN lcb_medium_55 template (no --reasoning-budget) and truncated to
# 44/55 = 80.00%. This re-runs on lcb_medium_55_v4 (forces --reasoning-budget
# 12288) under --sampler recommended for an untruncated vendor_base anchor.
# Pinned to GPU1 (free after vendor recovery); greedy anchor owns GPU0. Each is a
# single-bench --only run (no per-bench server relaunch) -> no co-location hazard.
# Lands as sibling lcb_medium_55_v4/ subdir under the vendorbase variant dir,
# preserving the original (truncated) lcb_medium_55/ cell.
set -uo pipefail
export OMK_ROOT=/srv/ml/repos/omnimergekit-canonical
export OMK_WS=/srv/ml/partc_b9700
export OMK_TOKENIZER=/srv/ml/google/gemma-4-26B-A4B-it
export LLAMA_BIN=/mnt/sdc/ml/llama.cpp-b9700/build/bin
export OMK_GPUS=1
export OMK_GPU_WAIT_S=300
cd "$OMK_ROOT/eval" || { echo "FATAL cannot cd $OMK_ROOT/eval"; exit 7; }
echo "[lcb-v4-vendor $(date '+%T %Z')] launching --only lcb_medium_55_v4, sampler=recommended, OMK_GPUS=1 port 8090"
./eval_suite_llama.sh \
  --variant 128e_b9700_vendorbase \
  --gguf /mnt/sdc/ml/eval_gguf/128e-Q6_K.gguf \
  --port 8090 \
  --sampler recommended --sampler-profile gemma-4 \
  --only lcb_medium_55_v4
echo "[lcb-v4-vendor $(date '+%T %Z')] LCB_V4_VENDOR_DONE rc=$?"
