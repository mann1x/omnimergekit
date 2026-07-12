#!/bin/bash
# Smoke ifeval_100 against cell 1 (a2eac_calibonly) with new llama_parallel=8.
# Lets it run ~10 questions then we check: VRAM, token stats, p50 chars.
set -uo pipefail
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH

CELL=a2eac_calibonly-62e-fc15_25-p8-s1_0p1_20
GGUF_DIR=/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-calibonly-it-GGUF
GGUF=$GGUF_DIR/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-calibonly-it-Q6_K.gguf
RES=/srv/ml/eval_results_tracks_2_3
TPL=ifeval_100
PORT=8195

LIVE=$RES/$TPL/$CELL
if [ -d "$LIVE" ]; then
    mv "$LIVE" "${LIVE}_p2_validation_$(date +%Y%m%d_%H%M%S)"
fi

LOG=/srv/ml/logs/t172_smoke_p8_$(date +%Y%m%d_%H%M%S).log
echo "log: $LOG"

nohup /srv/ml/envs/envs/omnimergekit/bin/python \
    /srv/ml/repos/omnimergekit/eval/omk_eval.py \
    --model "$GGUF" \
    --tokenizer "$GGUF_DIR" \
    --template $TPL \
    --backend llama \
    --served-name $CELL \
    --port $PORT \
    --results-dir $RES \
    > "$LOG" 2>&1 &
EVAL_PID=$!
disown
echo "EVAL_PID=$EVAL_PID"
echo "$EVAL_PID" > /tmp/t172_smoke_p8.pid
echo "$LOG" > /tmp/t172_smoke_p8.log
