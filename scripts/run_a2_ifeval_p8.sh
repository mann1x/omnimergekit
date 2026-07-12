#!/bin/bash
set -uo pipefail
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH
PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
RES=/srv/ml/eval_results_tracks_2_3
Q6=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-pes120-it-Q6_K.gguf
TOK=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it
pkill -KILL -f "llama-server.*--port 8197" 2>/dev/null; sleep 1
CUDA_VISIBLE_DEVICES=0 $PY $OMK --model "$Q6" --tokenizer "$TOK" --template ifeval_100 \
  --backend llama --port 8197 --served-name a2-anchor-p8 --results-dir "$RES"
