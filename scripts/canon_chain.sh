#!/usr/bin/env bash
# Generic canonical chain. Args: GPU PORT Q6 TOK NAME RES GATE_LOG GATE_MARK tpl1 tpl2 ...
# If GATE_MARK non-empty, wait until it appears in GATE_LOG (upstream chain freed the GPU)
# before booting. Server-death watchdog kills the chain if /v1/models is down >6min.
set -uo pipefail
GPU="$1"; PORT="$2"; Q6="$3"; TOK="$4"; NAME="$5"; RES="$6"; GATE_LOG="$7"; GATE_MARK="$8"; shift 8
TEMPLATES=("$@")
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH
PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
export CUDA_VISIBLE_DEVICES="$GPU"
PGID=$$
L(){ echo "[$NAME-g$GPU $(date -u +%H:%M:%S)] $*"; }

if [ -n "$GATE_MARK" ]; then
  L "GATED waiting for \"$GATE_MARK\" in $(basename "$GATE_LOG")"
  while ! grep -q "$GATE_MARK" "$GATE_LOG" 2>/dev/null; do sleep 30; done
  L "GATE released; settling 15s for server teardown"; sleep 15
fi

watchdog(){
  local down=0
  while kill -0 "$PGID" 2>/dev/null; do
    if curl -sf -m 3 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then down=0
    else down=$((down+1)); if [ "$down" -ge 12 ]; then
        L "WATCHDOG_KILL server :$PORT down ${down}x (~6min) — killing chain"
        kill -TERM -"$PGID" 2>/dev/null; sleep 5; kill -KILL -"$PGID" 2>/dev/null; return
      fi; fi
    sleep 30
  done
}
watchdog & WD=$!

L "START gpu=$GPU port=$PORT model=$NAME templates=${TEMPLATES[*]}"
for tpl in "${TEMPLATES[@]}"; do
  L ">>> eval $tpl"
  "$PY" "$OMK" --model "$Q6" --tokenizer "$TOK" --template "$tpl" --backend llama \
    --served-name "$NAME" --results-dir "$RES" --port "$PORT" 2>&1 | tail -20
  L "<<< $tpl rc=${PIPESTATUS[0]}"
done
kill "$WD" 2>/dev/null
L "CANON_CHAIN_${NAME}_G${GPU}_DONE"
