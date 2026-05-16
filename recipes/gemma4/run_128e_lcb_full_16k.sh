#!/usr/bin/env bash
# Symmetric 128e LCB-medium full rerun @ --max-tokens 16384 so the v4-vs-128e
# comparison stays apples-to-apples once v4 is rerun at 16384.
#
# 128e @ 8192 baseline: 87.27% (48/55), 2/55 cap-hits.
#   - 3793: borderline (3x "else:" repetition) — likely recovers at 16384
#   - 3805: severe loop (65x identical comment) — will still fail
#
# Runs AFTER the v4 16k rerun (run_v4_lcb_full_16k.sh, pid arg) finishes,
# so we don't fight for the 3090.
set -euo pipefail
WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
LOGS=$WS/logs
GGUF=$WS/google/gemma-4-26B-A4B-it-Q6_K.gguf
NAME=gemma4-A4B-128e-Q6K
PORT=8099
RESULTS=$WS/eval_results_128e_full_v2/lcb_16k
mkdir -p "$RESULTS"

eval "$(/root/anaconda3/bin/conda shell.bash hook)" && conda activate omnimergekit
ts() { date +%Y%m%d_%H%M%S; }
log() { echo "[$(date -Iseconds)] $*"; }

# Wait for v4 16k rerun to finish (pid 3191637 owns the chain wait)
PRED_PID=${1:-3191637}
log "waiting for predecessor pid=$PRED_PID ..."
while kill -0 "$PRED_PID" 2>/dev/null; do sleep 60; done
log "  predecessor exited; settling 30 s ..."
sleep 30
pkill -KILL -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 5

# Server: identical settings to the 8192 128e run (run_128e_lcb_full_v2.sh)
SLOG=$LOGS/server_128e_lcb_16k_$(ts).log
log "starting llama-server (128e chat profile) ..."
/opt/llama.cpp/build/bin/llama-server -m "$GGUF" --port $PORT \
    -c 32768 -t 12 -ngl 99 --no-warmup --parallel 2 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --temp 0 --top-p 1 --top-k 0 --seed 42 \
    --jinja --reasoning off > "$SLOG" 2>&1 &
SPID=$!
disown
log "  pid=$SPID log=$SLOG"
ready=0
for i in $(seq 1 60); do
    if curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        log "  ready"; ready=1; break
    fi
    # If server process died, no point continuing.
    if ! kill -0 "$SPID" 2>/dev/null; then
        log "  ABORT: llama-server pid=$SPID died during startup. Tail of $SLOG:"
        tail -25 "$SLOG" | sed 's/^/    /'
        exit 2
    fi
    sleep 2
done
if [[ $ready -ne 1 ]]; then
    log "  ABORT: server never responded on port $PORT after 120 s. Tail of $SLOG:"
    tail -25 "$SLOG" | sed 's/^/    /'
    kill -KILL $SPID 2>/dev/null || true
    exit 3
fi

T0=$(date +%s)
log "running 128e LCB-medium full @ max_tokens=16384 ..."
python3 /shared/dev/omnimergekit/eval/lcb/lcb_llama_server.py \
    --name "$NAME" --base-url "http://localhost:$PORT" \
    --limit 999 --max-tokens 16384 \
    --output "$RESULTS/${NAME}_lcb_full_16k.json" \
    > "$LOGS/128e_lcb_16k_$(ts).log" 2>&1 || true
log "  LCB wall=$(($(date +%s)-T0))s"

kill -TERM $SPID 2>/dev/null || true
sleep 2; kill -KILL $SPID 2>/dev/null || true
pkill -KILL -f "llama-server.*--port $PORT" 2>/dev/null || true

python3 - <<EOF
import json
new = json.load(open("$RESULTS/${NAME}_lcb_full_16k.json"))
print(f"\n=== 128e LCB-medium full @ 16384 ===")
print(f"  pass@1 = {new['pass_at_1']*100:.2f}%  ({new['n_pass']}/{new['n']})")
new_p = "$RESULTS/${NAME}_lcb_full_16k.samples.jsonl"
try:
    samples = {json.loads(l)['task_id']: json.loads(l) for l in open(new_p)}
    caps = sum(1 for r in samples.values() if r.get('finish_reason')=='length')
    print(f"  finish_reason=length: {caps}/{len(samples)}")
    old = {json.loads(l)['task_id']: json.loads(l) for l in open("$WS/eval_results_128e_full_v2/lcb/${NAME}_lcb_full.samples.jsonl")}
    recov = [t for t in samples if (not old.get(t,{}).get('passed')) and samples[t].get('passed')]
    regr  = [t for t in samples if old.get(t,{}).get('passed') and not samples[t].get('passed')]
    print(f"  recovered (8192 fail → 16384 pass): {len(recov)} -> {recov}")
    print(f"  new regressions (expect 0):         {len(regr)} -> {regr}")
except Exception as e:
    print(f"  diff err: {e}")
EOF

log "DONE — results in $RESULTS"
