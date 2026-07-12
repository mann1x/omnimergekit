#!/bin/bash
# Audit T172 combo cell: shared 0.833 + pes 1.44 (decoded A2 recipe)
set -uo pipefail
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
AUDIT=$BM/scripts/audit_full_bench.py
RES_FULL=$BM/eval_results_tracks_2_3
COMBO=gemma-4-A4B-62e-fc15_25-p8-shared083-pes144-sf-it
GGUF_DIR=/mnt/sdc/ml/google/${COMBO}-GGUF
Q6=$GGUF_DIR/${COMBO}-Q6_K.gguf
SERVED=$COMBO
BASELINE=a2-62e-fc15_25-p8-s1_0p1_20

export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"

TS=$(date +%Y%m%d_%H%M%S)
LOG=$BM/logs/t172/audit_combo_${TS}.log
exec > >(tee "$LOG") 2>&1

echo "[$(date -Iseconds)] === T172 combo audit: shared 0.833 × pes 1.44 (decoded A2) ==="
echo "  Q6      : $Q6"
echo "  served  : $SERVED"
echo "  baseline: $BASELINE"
echo

for tpl in humanevalplus_full ifeval_100 multipl_e_100; do
    sd=$RES_FULL/$tpl/$SERVED
    [ -f "$sd/summary.json" ] && rm -rf "$sd"
    tlim=1800
    [ "$tpl" = "multipl_e_100" ] && tlim=3600
    echo "[$(date -Iseconds)] >>> $tpl"
    timeout --kill-after=10 --signal=KILL $tlim \
        "$PY" "$OMK" --model "$Q6" --tokenizer "$GGUF_DIR" --template $tpl --backend llama \
        --served-name "$SERVED" --results-dir "$RES_FULL" 2>&1 | tail -8
    echo "[$(date -Iseconds)] <<< $tpl rc=${PIPESTATUS[0]}"
    if pgrep -f "llama-server.*--port 8195" >/dev/null; then
        pkill -KILL -f "llama-server.*--port 8195"; sleep 2
    fi
done

echo
echo "[$(date -Iseconds)] === AUDIT phase ==="
for tpl in humanevalplus_full ifeval_100 multipl_e_100; do
    "$PY" "$AUDIT" $tpl "$SERVED" "$BASELINE" 2>&1
done

echo "[$(date -Iseconds)] === combo audit DONE ==="
