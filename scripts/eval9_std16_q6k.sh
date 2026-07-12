#!/usr/bin/env bash
# eval9_v8_q6k.sh — full canonical 9-bench eval of v8 (fkbroad-soft2) imat-Q6_K
# on llama.cpp b9700, GREEDY (apples-to-apples vs the published v7-coder card).
# Dual-GPU: shard A on GPU0:8092, shard B on GPU1:8093, run in parallel.
# Variant std16_fk16_q6k; LCB uses the budgeted _v4 template.
set -uo pipefail
export OMK_ROOT=/srv/ml/repos/omnimergekit-canonical
export OMK_WS=/srv/ml/std16_q6k_full
export OMK_TOKENIZER=/srv/ml/google/gemma-4-26B-A4B-it
export LLAMA_BIN=/mnt/sdc/ml/llama.cpp-b9700/build/bin
GGUF=/mnt/sdc/ml/t223_fk/STD16-imatq6.gguf
cd "$OMK_ROOT/eval" || { echo "FATAL cannot cd $OMK_ROOT/eval"; exit 7; }

run_shard () {
  local gpu=$1 port=$2 only=$3 tag=$4
  OMK_GPUS=$gpu OMK_GPU_WAIT_S=300 ./eval_suite_llama.sh \
    --variant std16_fk16_q6k \
    --gguf "$GGUF" \
    --port "$port" \
    --only "$only" \
    > "/srv/ml/scripts/eval9_std16_q6k_${tag}.run.log" 2>&1
  echo "[eval9-std16 $(date '+%T %Z')] SHARD_${tag}_DONE rc=$?"
}

echo "[eval9-std16 $(date '+%T %Z')] launch dual-GPU 9-bench greedy b9700; gguf=$(stat -c %s "$GGUF" | numfmt --to=iec)"
run_shard 0 8092 "gpqa_diamond_full,gsm8k_100,math500_100,aime_30" A &
run_shard 1 8093 "arc_challenge_full,ifeval_100,humaneval_full,humanevalplus_full,lcb_medium_55_v4" B &
wait
echo "[eval9-std16 $(date '+%T %Z')] EVAL9_V8_ALL_DONE rc=0"
