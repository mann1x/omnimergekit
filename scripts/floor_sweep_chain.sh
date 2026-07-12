#!/usr/bin/env bash
# floor_sweep_chain.sh GPU PORT VARIANT GATE_LOG GATE_MARK
# Dual gate: wait GPU free (GATE_MARK in GATE_LOG) AND model built (${Q6}.BUILD_DONE).
# Then eval gpqa_diamond_full + lcb_medium_55_v4. Server-death watchdog.
set -uo pipefail
GPU="$1"; PORT="$2"; VAR="$3"; GATE_LOG="$4"; GATE_MARK="$5"
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH
PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
GG=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-$VAR-it-GGUF
Q6=$GG/gemma-4-A4B-98e-v7-coder-$VAR-it-Q6_K.gguf
TOK=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-$VAR-it
NAME=v7coder-$VAR-q6k
RES=/srv/ml/eval_results_v7coder_$VAR
export CUDA_VISIBLE_DEVICES="$GPU"
PGID=$$
L(){ echo "[fsweep-$VAR-g$GPU $(date -u +%H:%M:%S)] $*"; }
mkdir -p "$RES"
L "GATED on $GATE_MARK (GPU) + $(basename "$Q6").BUILD_DONE (model)"
while ! grep -q "$GATE_MARK" "$GATE_LOG" 2>/dev/null; do sleep 30; done
L "GPU gate released"
while [ ! -f "${Q6}.BUILD_DONE" ]; do sleep 30; done
L "build gate released; settle 15s"; sleep 15
watchdog(){ local d=0; while kill -0 "$PGID" 2>/dev/null; do
  if curl -sf -m 3 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then d=0
  else d=$((d+1)); [ "$d" -ge 12 ] && { L "WATCHDOG_KILL :$PORT down ~6min"; kill -TERM -"$PGID" 2>/dev/null; sleep 5; kill -KILL -"$PGID" 2>/dev/null; return; }; fi
  sleep 30; done; }
watchdog & WD=$!
L "START $VAR gpu=$GPU port=$PORT"
for tpl in gpqa_diamond_full lcb_medium_55_v4; do
  L ">>> eval $tpl"
  "$PY" "$OMK" --model "$Q6" --tokenizer "$TOK" --template "$tpl" --backend llama \
    --served-name "$NAME" --results-dir "$RES" --port "$PORT" 2>&1 | tail -20
  L "<<< $tpl rc=${PIPESTATUS[0]}"
done
kill "$WD" 2>/dev/null
L "FLOOR_SWEEP_${VAR}_DONE"
