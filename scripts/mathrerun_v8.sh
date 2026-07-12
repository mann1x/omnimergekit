#!/usr/bin/env bash
set -uo pipefail
export OMK_ROOT=/srv/ml/repos/omnimergekit-canonical
export LLAMA_BIN=/mnt/sdc/ml/llama.cpp-b9700/build/bin
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
export LLAMA_EXTRA="--jinja --reasoning off"
GGUF=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-fkbroad-soft2-imat-Q6_K.gguf
TOK=/srv/ml/google/gemma-4-26B-A4B-it
run () { local gpu=$1 port=$2 tmpl=$3
  OMK_GPUS=$gpu OMK_GPU_WAIT_S=60 /root/anaconda3/envs/omnimergekit/bin/python "$OMK_ROOT/eval/omk_eval.py" \
    --backend llama --template "$tmpl" --quant q6_k \
    --model "$GGUF" --tokenizer "$TOK" \
    --served-name v8rr_q6k --port "$port" \
    --results-dir /srv/ml/v8_mathrerun \
    > "/srv/ml/scripts/mathrerun_v8_${tmpl}.run.log" 2>&1
  echo "[mathrerun $tmpl done rc=$? $(date +%T)]"; }
run 0 8092 gsm8k_100 &
run 1 8093 math500_100 &
wait
echo "[mathrerun ALL_DONE $(date +%T)]"
