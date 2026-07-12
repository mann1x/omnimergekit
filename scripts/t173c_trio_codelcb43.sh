#!/bin/bash
set -uo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
AUDIT=/srv/ml/scripts/audit_full_bench.py
LOOPCK=/srv/ml/repos/omnimergekit/scripts/loop_precision_check.py
RES=/srv/ml/eval_results_tracks_2_3
WORK=/mnt/sdc/ml/google
PRISTINE=gemma-4-A4B-62e-fc15_25-p8-pristine-it
NAME=gemma-4-A4B-62e-fc15_25-p8-codelcb43-it
TLIM=9000
export PATH="/srv/ml/envs/envs/omnimergekit/bin:${PATH:-/usr/bin:/bin}"
export LD_LIBRARY_PATH="/srv/ml/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"
LOG_DIR=/srv/ml/logs/t173; mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
exec > >(tee "$LOG_DIR/t173c_trio_${TS}.log") 2>&1

eval_one() {
    local bench=$1 gpu=$2 port=$3
    local gdir="$WORK/$NAME-GGUF" q6="$WORK/$NAME-GGUF/${NAME}-Q6_K.gguf"
    local sd="$RES/$bench/$NAME"
    [ -d "$sd" ] && rm -rf "$sd"
    pkill -KILL -f "llama-server.*--port $port" 2>/dev/null; sleep 2
    echo "[eval gpu$gpu $(date +%H:%M:%S)] $bench $NAME"
    CUDA_VISIBLE_DEVICES=$gpu timeout --kill-after=10 --signal=KILL "$TLIM" \
        "$PY" "$OMK" --model "$q6" --tokenizer "$gdir" --template "$bench" \
        --backend llama --port "$port" --served-name "$NAME" --results-dir "$RES" 2>&1 \
        | sed "s/^/[gpu$gpu $bench] /" | tail -8
    pkill -KILL -f "llama-server.*--port $port" 2>/dev/null; sleep 2
    echo "[eval gpu$gpu $(date +%H:%M:%S)] DONE $bench"
}

echo "==================== T173c TRIO codelcb43 (dual GPU) ===================="
eval_one humanevalplus_full 0 8295 & pA=$!
eval_one multipl_e_100      1 8296 & pB=$!
wait "$pA" "$pB"

echo "==================== T173c TRIO AUDIT vs pristine ===================="
for b in humanevalplus_full multipl_e_100; do
    "$PY" "$AUDIT" "$b" "$PRISTINE"  "$PRISTINE" 2>/dev/null | grep "^AUDIT" || true
    "$PY" "$AUDIT" "$b" "$NAME" "$PRISTINE" 2>/dev/null | grep "^AUDIT" || echo "AUDIT_FAIL $b $NAME"
    "$PY" "$LOOPCK" "$b" "$NAME" 2>/dev/null | grep -E "loop-flagged|TOTAL" || true
done
echo "==================== T173c TRIO DONE $(date +%H:%M:%S) ===================="
