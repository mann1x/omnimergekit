#!/usr/bin/env bash
# Local 128e LCB-medium full run for v4 headline parity.
# Mirrors the pod chat-profile server settings.
set -euo pipefail
WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
LOGS=$WS/logs
GGUF=$WS/google/gemma-4-26B-A4B-it-Q6_K.gguf
NAME=gemma4-A4B-128e-Q6K
PORT=8099
RESULTS=$WS/eval_results_128e_full
mkdir -p $RESULTS/lcb

eval "$(/root/anaconda3/bin/conda shell.bash hook)" && conda activate omnimergekit
ts() { date +%Y%m%d_%H%M%S; }

SLOG=$LOGS/server_128e_lcb_full_$(ts).log
/opt/llama.cpp/build/bin/llama-server -m "$GGUF" --port $PORT \
    -c 16384 -t 12 -ngl 99 --no-warmup --parallel 2 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --temp 0 --top-p 1 --top-k 0 --seed 42 \
    --jinja --reasoning off > "$SLOG" 2>&1 &
SPID=$!
disown
echo "  llama-server pid=$SPID log=$SLOG"
for i in $(seq 1 60); do
    curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "  ready"; break; }
    sleep 2
done

T0=$(date +%s)
python3 /shared/dev/omnimergekit/eval/lcb/lcb_llama_server.py \
    --name "$NAME" --base-url "http://localhost:$PORT" \
    --limit 999 --output $RESULTS/lcb/${NAME}_lcb_full.json \
    > $LOGS/full_128e_lcb_$(ts).log 2>&1 || true
echo "LCB-full wall=$(($(date +%s)-T0))s"

kill -TERM $SPID 2>/dev/null || true
sleep 2
kill -KILL $SPID 2>/dev/null || true
pkill -KILL -f "llama-server.*-port $PORT" 2>/dev/null || true

echo DONE.
