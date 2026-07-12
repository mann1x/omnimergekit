#!/usr/bin/env bash
set -uo pipefail
BM=/srv/ml; PY=$BM/envs/envs/omnimergekit/bin/python
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"
GPU=$1; PORT=$2; NAME=$3; Q6=$4
export CUDA_VISIBLE_DEVICES=$GPU
TOK=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-floor-it
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
RES=$BM/eval_results_v7coder_pes
echo "[lcb-$NAME $(date -u +%H:%M:%S)] >>> lcb_medium_55_v4 GPU$GPU port$PORT"
"$PY" "$OMK" --model "$Q6" --tokenizer "$TOK" --template lcb_medium_55_v4 --backend llama --port "$PORT" \
  --served-name "$NAME" --results-dir "$RES" 2>&1 | tail -18
s=$("$PY" -c "import json;print(json.load(open(\"$RES/lcb_medium_55_v4/$NAME/summary.json\")).get(\"score\"))" 2>/dev/null)
echo "[lcb-$NAME $(date -u +%H:%M:%S)] LCB_DONE $NAME score=$s"
