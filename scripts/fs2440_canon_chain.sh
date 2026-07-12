#!/usr/bin/env bash
# fs2440 canonical-suite chain (one GPU, one port). Args: GPU PORT tpl1 tpl2 ...
# Server-death watchdog: if /v1/models is down >6min continuously while the chain
# runs, kill the chain (stops the lm-eval "Retrying in" grind on a dead server).
set -uo pipefail
GPU="$1"; PORT="$2"; shift 2; TEMPLATES=("$@")
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH   # bug-022: lm-eval on PATH
PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
Q6=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it-GGUF/gemma-4-A4B-98e-v7-coder-fs2440-it-Q6_K.gguf
TOK=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it
NAME=v7coder-fs2440-q6k
RES=/srv/ml/eval_results_v7coder_fs2440
export CUDA_VISIBLE_DEVICES="$GPU"
PGID=$$
L(){ echo "[canon-g$GPU $(date -u +%H:%M:%S)] $*"; }

watchdog(){
  local down=0
  while kill -0 "$PGID" 2>/dev/null; do
    if curl -sf -m 3 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
      down=0
    else
      down=$((down+1))
      if [ "$down" -ge 12 ]; then
        L "WATCHDOG_KILL server :$PORT down ${down}x (~6min) — killing chain to stop retry grind"
        kill -TERM -"$PGID" 2>/dev/null; sleep 5; kill -KILL -"$PGID" 2>/dev/null
        return
      fi
    fi
    sleep 30
  done
}
watchdog & WD=$!

L "START gpu=$GPU port=$PORT templates=${TEMPLATES[*]}"
for tpl in "${TEMPLATES[@]}"; do
  L ">>> eval $tpl"
  "$PY" "$OMK" --model "$Q6" --tokenizer "$TOK" --template "$tpl" --backend llama \
    --served-name "$NAME" --results-dir "$RES" --port "$PORT" 2>&1 | tail -20
  L "<<< $tpl rc=${PIPESTATUS[0]}"
done
kill "$WD" 2>/dev/null
L "CANON_CHAIN_G${GPU}_DONE"
