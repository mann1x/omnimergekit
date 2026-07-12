#!/bin/bash
# T172.5: re-run combo IFEval-100 with the patched template (parallel=2 via
# omk_eval safety clamp). No --parallel flag needed — the canonical
# template now sets llama_parallel=2 and omk_eval would clamp it anyway.
set -uo pipefail

PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
AUDIT=/srv/ml/scripts/audit_full_bench.py

export PATH="/srv/ml/envs/envs/omnimergekit/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export LD_LIBRARY_PATH="/srv/ml/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"

VARIANT=gemma-4-A4B-62e-fc15_25-p8-shared083-pes144-sf-it
Q6=/mnt/sdc/ml/google/${VARIANT}-GGUF/${VARIANT}-Q6_K.gguf
TOK=/mnt/sdc/ml/google/${VARIANT}-GGUF
RES=/srv/ml/eval_results_tracks_2_3
BASELINE=a2-62e-fc15_25-p8-s1_0p1_20

OLD=$RES/ifeval_100/$VARIANT
if [ -d "$OLD" ]; then
    BAK=$RES/ifeval_100/${VARIANT}_p8_OLD
    [ -d "$BAK" ] && rm -rf "$BAK"
    mv "$OLD" "$BAK"
fi

LOG_DIR=/srv/ml/logs/t172
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/t172_5_combo_validate_${TS}.log
exec > >(tee "$LOG") 2>&1

echo "[$(date -Iseconds)] === T172.5 combo IFEval-100 re-run with patched omk_eval ==="
echo "  Q6      : $Q6"
echo "  served  : $VARIANT"
echo "  expect  : parallel=2 (template canonical), per-slot ctx=16384, score → ~0.87"

pkill -KILL -f "llama-server.*--port 8195" 2>/dev/null
sleep 2

timeout --kill-after=10 --signal=KILL 5400 \
    "$PY" "$OMK" --model "$Q6" --tokenizer "$TOK" --template ifeval_100 \
    --backend llama \
    --served-name "$VARIANT" --results-dir "$RES" 2>&1 | tail -40

if pgrep -f "llama-server.*--port 8195" >/dev/null; then
    pkill -KILL -f "llama-server.*--port 8195"; sleep 2
fi

# Confirm slot config from new server.log
echo
echo "=== verify per-slot config ==="
NEW_LOG=$RES/ifeval_100/$VARIANT/server.log
if [ -f "$NEW_LOG" ]; then
    grep -E "n_slots|new slot, n_ctx" "$NEW_LOG" | head -5
fi

echo
echo "[$(date -Iseconds)] === AUDIT ==="
"$PY" "$AUDIT" ifeval_100 "$VARIANT" "$BASELINE" 2>&1 | grep '^AUDIT' || echo "AUDIT_FAIL"
echo
jq '{score, samples_processed, "metric": .metric}' "$RES/ifeval_100/$VARIANT/summary.json" 2>/dev/null
echo
echo "[$(date -Iseconds)] === DONE ==="
