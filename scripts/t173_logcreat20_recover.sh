#!/bin/bash
set -uo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
WORK=/mnt/sdc/ml/google
RES=/srv/ml/eval_results_tracks_2_3
BENCH=ifeval_100
TLIM=5400
name=gemma-4-A4B-62e-fc15_25-p8-logcreat20-it
gdir="$WORK/$name-GGUF"
q6="$gdir/${name}-Q6_K.gguf"
sd="$RES/$BENCH/$name"
export PATH="/srv/ml/envs/envs/omnimergekit/bin:${PATH:-/usr/bin:/bin}"
export LD_LIBRARY_PATH="/srv/ml/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"
[ -d "$sd" ] && rm -rf "$sd"
pkill -KILL -f "llama-server.*--port 8295" 2>/dev/null; sleep 2
echo "[recover $(date +%H:%M:%S)] logcreat20 eval start gpu0:8295"
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=10 --signal=KILL "$TLIM" \
    "$PY" "$OMK" --model "$q6" --tokenizer "$gdir" --template "$BENCH" \
    --backend llama --port 8295 --served-name "$name" --results-dir "$RES" 2>&1 | tail -8
pkill -KILL -f "llama-server.*--port 8295" 2>/dev/null; sleep 2
echo "[recover $(date +%H:%M:%S)] DONE logcreat20"
