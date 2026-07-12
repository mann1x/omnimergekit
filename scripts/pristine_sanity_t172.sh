#!/bin/bash
# T172 pristine baseline sanity audit — run HE+/IF/MPE on pristine Q6_K, then audit.
# Must agree with A2 within ±2pp on each bench. If not, pristine build is wrong.
set -uo pipefail
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
AUDIT=$BM/scripts/audit_full_bench.py
RES_FULL=$BM/eval_results_tracks_2_3
PRISTINE_DIR=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pristine-it-GGUF
Q6=$PRISTINE_DIR/gemma-4-A4B-62e-fc15_25-p8-pristine-it-Q6_K.gguf
SERVED=gemma-4-A4B-62e-fc15_25-p8-pristine-it
BASELINE=a2-62e-fc15_25-p8-s1_0p1_20

export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"

TS=$(date +%Y%m%d_%H%M%S)
LOG=$BM/logs/t172/pristine_sanity_${TS}.log
mkdir -p "$BM/logs/t172"
exec > >(tee "$LOG") 2>&1

echo "[$(date -Iseconds)] === T172 pristine sanity audit ==="
echo "  Q6      : $Q6"
echo "  tok dir : $PRISTINE_DIR"
echo "  served  : $SERVED"
echo "  baseline: $BASELINE"
echo "  log     : $LOG"
echo

for tpl in humanevalplus_full ifeval_100 multipl_e_100; do
    sd="$RES_FULL/$tpl/$SERVED"
    if [ -f "$sd/summary.json" ]; then
        score=$(jq -r ".score // empty" "$sd/summary.json")
        if [ -n "$score" ] && [ "$score" != "null" ]; then
            echo "[skip] $tpl already done (score=$score)"
            continue
        fi
        rm -rf "$sd"
    fi
    tlim=1800
    [ "$tpl" = "multipl_e_100" ] && tlim=3600
    echo "[$(date -Iseconds)] >>> $tpl"
    timeout --kill-after=10 --signal=KILL $tlim \
        "$PY" "$OMK" --model "$Q6" --tokenizer "$PRISTINE_DIR" --template $tpl --backend llama \
        --served-name "$SERVED" --results-dir "$RES_FULL" 2>&1 | tail -10
    rc=${PIPESTATUS[0]}
    echo "[$(date -Iseconds)] <<< $tpl rc=$rc"
    if pgrep -f "llama-server.*--port 8195" >/dev/null; then
        echo "[cleanup] killing leftover llama-server on 8195"
        pkill -KILL -f "llama-server.*--port 8195"
        sleep 2
    fi
done

echo
echo "[$(date -Iseconds)] === AUDIT phase ==="
for tpl in humanevalplus_full ifeval_100 multipl_e_100; do
    "$PY" "$AUDIT" $tpl "$SERVED" "$BASELINE" 2>&1
done

echo
echo "[$(date -Iseconds)] === pristine sanity DONE ==="
