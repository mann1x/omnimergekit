#!/usr/bin/env bash
set -uo pipefail
export OMK_ROOT=/srv/ml/repos/omnimergekit-canonical
export LLAMA_BIN=/mnt/sdc/ml/llama.cpp-b9700/build/bin
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
export LLAMA_EXTRA="--reasoning-format none --reasoning-budget 12288"
GGUF=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-fkbroad-soft2-imat-Q6_K.gguf
OMK_GPUS=0 OMK_GPU_WAIT_S=120 /root/anaconda3/envs/omnimergekit/bin/python "$OMK_ROOT/eval/omk_eval.py" \
  --backend llama --template gsm8k_100 --quant q6_k \
  --model "$GGUF" --tokenizer /srv/ml/google/gemma-4-26B-A4B-it \
  --served-name v8probe_q6k --port 8092 \
  --results-dir /srv/ml/v8_gsm8k_probe --metadata n=10 \
  > /srv/ml/scripts/gsm8k_v8_probe.run.log 2>&1
echo "[probe done rc=$?]"
