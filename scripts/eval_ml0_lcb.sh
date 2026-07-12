#!/usr/bin/env bash
set -uo pipefail
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=1
Q6=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-it-GGUF/gemma-4-A4B-98e-v7-coder-it-Q6_K.gguf
TOK=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-it
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
RES=$BM/eval_results_v7coder
echo "[ml0-lcb $(date -u +%H:%M:%S)] >>> lcb_medium_55_v4 on ml0 (GPU1)"
"$PY" "$OMK" --model "$Q6" --tokenizer "$TOK" \
  --template lcb_medium_55_v4 --backend llama \
  --served-name v7coder-C6v3lcb-q6k --results-dir "$RES" 2>&1 | tail -20
s=$("$PY" -c "import json;print(json.load(open(\"$RES/lcb_medium_55_v4/v7coder-C6v3lcb-q6k/summary.json\")).get(\"score\"))" 2>/dev/null)
echo "[ml0-lcb $(date -u +%H:%M:%S)] ML0_LCB_DONE score=$s"
