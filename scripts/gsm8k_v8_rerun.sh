#!/usr/bin/env bash
# Clean GSM8K-100 re-run for v8 after the 9-bench suite finishes (peg-parser crash at q92).
# Resumes from the suite sqlite cache; fresh server. Pin a single GPU, wait if busy.
set -uo pipefail
export OMK_ROOT=/srv/ml/repos/omnimergekit-canonical
export OMK_TOKENIZER=/srv/ml/google/gemma-4-26B-A4B-it
export LLAMA_BIN=/mnt/sdc/ml/llama.cpp-b9700/build/bin
GGUF=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-fkbroad-soft2-imat-Q6_K.gguf
GPU="${1:-0}"; PORT="${2:-8092}"
cd "$OMK_ROOT/eval" || exit 9
OMK_GPUS="$GPU" OMK_GPU_WAIT_S=600 /root/anaconda3/envs/omnimergekit/bin/python "$OMK_ROOT/eval/omk_eval.py" \
  --backend llama --template gsm8k_100 --quant q6_k \
  --model "$GGUF" --tokenizer "$OMK_TOKENIZER" \
  --served-name v8_fkbroad_soft2_q6k_q6k --port "$PORT" \
  --results-dir /srv/ml/v8_q6k_full/eval_results_llama_suite/v8_fkbroad_soft2_q6k \
  > /srv/ml/scripts/gsm8k_v8_rerun.run.log 2>&1
echo "[gsm8k-v8-rerun $(date +%T\ %Z)] DONE rc=$?"
