#!/usr/bin/env bash
set -uo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
GGUF=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF/gemma-4-A4B-98e-v7-coder-it-Q6_K.gguf
TOK=/mnt/sdc/ml/t223_fk/STD16-bf16
WORK=/mnt/sdc/ml/std16_gate/body_fill
LOG=$WORK/mpe100_gpu1.log
exec >>"$LOG" 2>&1
echo "==== STD16 MPE-100 greedy GPU1 start $(date -u +%T) ===="
CUDA_VISIBLE_DEVICES=1 "$PY" "$OMK" --model "$GGUF" --tokenizer "$TOK" --template multipl_e_100 \
  --backend llama --quant gguf --port 8092 --results-dir "$WORK/results_mpe" --served-name STD16
echo "STD16_MPE100_DONE rc=$?"
