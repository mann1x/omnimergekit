#!/bin/bash
# Resume combo IFEval-100 from sqlite cache (50/100 already done)
export PATH="/srv/ml/envs/envs/omnimergekit/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export LD_LIBRARY_PATH="/srv/ml/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"

PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
VARIANT=gemma-4-A4B-62e-fc15_25-p8-shared083-pes144-sf-it
Q6=/mnt/sdc/ml/google/${VARIANT}-GGUF/${VARIANT}-Q6_K.gguf
TOK=/mnt/sdc/ml/google/${VARIANT}-GGUF
RES=/srv/ml/eval_results_tracks_2_3

pkill -KILL -f "llama-server.*--port 8195" 2>/dev/null
sleep 2

timeout --kill-after=10 --signal=KILL 5400 \
    "$PY" "$OMK" --model "$Q6" --tokenizer "$TOK" --template ifeval_100 \
    --backend llama \
    --served-name "$VARIANT" --results-dir "$RES"
