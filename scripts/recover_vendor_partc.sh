#!/usr/bin/env bash
# recover_vendor_partc.sh — re-run the 8 Part C vendor benches that rc=8'd when
# the canonical-omk gpu_planner refused GPU1 (busy with the v8 LCB). Pins GPU1
# and WAITS (OMK_GPU_WAIT_S) for it to free when the v8 LCB finishes, then runs
# the 8 missing benches into the SAME vendor result dir. GPQA (already scored
# 72.73%) is excluded via --only. Greedy on GPU0 is untouched (OMK_GPUS=1).
set -uo pipefail
export OMK_ROOT=/srv/ml/repos/omnimergekit-canonical
export OMK_WS=/srv/ml/partc_b9700
export OMK_TOKENIZER=/srv/ml/google/gemma-4-26B-A4B-it
export LLAMA_BIN=/mnt/sdc/ml/llama.cpp-b9700/build/bin
export OMK_GPUS=1
export OMK_GPU_WAIT_S=2400
cd "$OMK_ROOT/eval" || { echo "FATAL cannot cd $OMK_ROOT/eval"; exit 7; }
echo "[recover-vendor $(date '+%T %Z')] launching --only 8 benches, OMK_GPUS=1 OMK_GPU_WAIT_S=2400"
./eval_suite_llama.sh \
  --variant 128e_b9700_vendorbase \
  --gguf /mnt/sdc/ml/eval_gguf/128e-Q6_K.gguf \
  --port 8090 \
  --sampler recommended --sampler-profile gemma-4 \
  --only gsm8k_100,math500_100,aime_30,arc_challenge_full,ifeval_100,humaneval_full,humanevalplus_full,lcb_medium_55
echo "[recover-vendor $(date '+%T %Z')] RECOVER_VENDOR_DONE rc=$?"
