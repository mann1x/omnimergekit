#!/bin/bash
# T172.8-dual: dual-server IFEval-100 re-eval, one llama-server per GPU.
#   Stream A: CUDA_VISIBLE_DEVICES=0  port 8195  -> even-index cells
#   Stream B: CUDA_VISIBLE_DEVICES=1  port 8196  -> odd-index cells
# Each server loads the FULL 17GB Q6_K on ONE 97GB Blackwell (no layer
# split) -> full single-GPU compute + 2-way stream concurrency.
# Cell 1 (a2eac_calibonly) already CLEAN @0.870 -> excluded (16 cells here).
#
# Sacred-data rule: each cell's original parallel=8/ctx=32768 result is moved
# to <dir>_p8_OLD exactly once and NEVER overwritten.
# Author: claude opus 4.8  2026-05-30 CEST
set -uo pipefail

PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
AUDIT=/srv/ml/scripts/audit_full_bench.py
RES=/srv/ml/eval_results_tracks_2_3
BASELINE=a2-62e-fc15_25-p8-s1_0p1_20
BENCH=ifeval_100
TLIM=5400

export PATH="/srv/ml/envs/envs/omnimergekit/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export LD_LIBRARY_PATH="/srv/ml/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"

LOG_DIR=/srv/ml/logs/t172
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/t172_rerun_dual_${TS}.log
SUMMARY=$LOG_DIR/t172_rerun_17cells_summary.tsv
exec > >(tee "$LOG") 2>&1

if [ ! -s "$SUMMARY" ]; then
    printf 'ts\tcell\tbench\tscore\tdelta_vs_a2\tflags\tverdict\n' > "$SUMMARY"
fi

# 16 cells (cell 1 a2eac_calibonly already CLEAN @0.870, excluded)
CELLS=(
"a2eac_9bench-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-9bench-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-9bench-it-Q6_K.gguf"
"a2eac_ifheavy-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-ifheavy-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-ifheavy-it-Q6_K.gguf"
"a2kdonly_calibonly-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-calibonly-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-calibonly-it-Q6_K.gguf"
"a2kdonly_9bench-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-9bench-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-9bench-it-Q6_K.gguf"
"a2kdonly_ifheavy-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-ifheavy-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-ifheavy-it-Q6_K.gguf"
"a2rkd_calibonly-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-calibonly-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-calibonly-it-Q6_K.gguf"
"a2rkd_9bench-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-9bench-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-9bench-it-Q6_K.gguf"
"a2rkd_ifheavy-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-ifheavy-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-ifheavy-it-Q6_K.gguf"
"gemma-4-A4B-62e-fc15_25-p8-pristine-it|/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pristine-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-pristine-it-Q6_K.gguf"
"gemma-4-A4B-62e-fc15_25-p8-shared110-it|/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-shared110-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-shared110-it-Q6_K.gguf"
"gemma-4-A4B-62e-fc15_25-p8-shared120-it|/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-shared120-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-shared120-it-Q6_K.gguf"
"gemma-4-A4B-62e-fc15_25-p8-shared130-it|/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-shared130-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-shared130-it-Q6_K.gguf"
"gemma-4-A4B-62e-fc15_25-p8-pes110-it|/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes110-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-pes110-it-Q6_K.gguf"
"gemma-4-A4B-62e-fc15_25-p8-pes120-it|/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-pes120-it-Q6_K.gguf"
"gemma-4-A4B-62e-fc15_25-p8-pes130-it|/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes130-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-pes130-it-Q6_K.gguf"
"pes1_10-62e-fc15_25-p8|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes1_10-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-pes1_10-it-Q6_K.gguf"
)

audit_emit() {
    local bench=$1 served=$2 gpu=$3
    local line
    line=$("$PY" "$AUDIT" "$bench" "$served" "$BASELINE" 2>/dev/null | grep '^AUDIT' || echo "AUDIT_FAIL")
    echo "[gpu$gpu] $line"
    local score delta flags verdict
    score=$(echo "$line" | sed -nE 's/.*\bscore=([^[:space:]]+).*/\1/p' | head -1)
    delta=$(echo "$line" | sed -nE 's/.*\bd=([^[:space:]]+).*/\1/p' | head -1)
    flags=$(echo "$line" | sed -nE 's/.*flags=\[([^]]*)\].*/\1/p' | head -1)
    verdict="MIXED"
    case "$flags" in
        "CLEAN") verdict="CLEAN" ;;
        *PARTIAL_BENCH*) verdict="PARTIAL" ;;
        *SAT_COLLAPSE*) verdict="SAT_COLLAPSE" ;;
        *KNOWLEDGE_SHIFT*) verdict="KNOWLEDGE_SHIFT" ;;
        *LEN_BLOAT*) verdict="LEN_BLOAT" ;;
    esac
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Iseconds)" "$served" "$bench" "${score:-?}" "${delta:-?}" "${flags:-?}" "$verdict" \
        >> "$SUMMARY"
}

bench_one() {
    local q6=$1 tok=$2 served=$3 port=$4 gpu=$5
    local sd="$RES/$BENCH/$served"
    if [ -d "$sd" ]; then
        if [ -d "${sd}_p8_OLD" ]; then
            rm -rf "$sd"
            echo "[gpu$gpu] discarded stale live ($served) — original safe in _p8_OLD"
        else
            mv "$sd" "${sd}_p8_OLD"
            echo "[gpu$gpu] preserved original → ${served}_p8_OLD"
        fi
    fi
    pkill -KILL -f "llama-server.*--port $port" 2>/dev/null
    sleep 2
    CUDA_VISIBLE_DEVICES=$gpu timeout --kill-after=10 --signal=KILL "$TLIM" \
        "$PY" "$OMK" --model "$q6" --tokenizer "$tok" --template "$BENCH" \
        --backend llama --port "$port" \
        --served-name "$served" --results-dir "$RES" 2>&1 | sed "s/^/[gpu$gpu] /" | tail -8
    if pgrep -f "llama-server.*--port $port" >/dev/null; then
        pkill -KILL -f "llama-server.*--port $port"; sleep 2
    fi
}

run_stream() {
    local gpu=$1 port=$2; shift 2
    local idxs=("$@")
    for i in "${idxs[@]}"; do
        local entry="${CELLS[$i]}"
        local served="${entry%%|*}"
        local q6="${entry#*|}"
        local tok; tok="$(dirname "$q6")"
        echo "[gpu$gpu] >>> cell[$i]: $served"
        if [ ! -f "$q6" ]; then
            echo "[gpu$gpu] SKIP missing $q6"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$served" "MISSING_Q6" "?" "?" "?" "SKIP" >> "$SUMMARY"
            continue
        fi
        bench_one "$q6" "$tok" "$served" "$port" "$gpu"
        audit_emit "$BENCH" "$served" "$gpu"
        echo "[gpu$gpu] [done] $served"
    done
    echo "[gpu$gpu] === STREAM COMPLETE ==="
}

A_IDX=(); B_IDX=()
for i in "${!CELLS[@]}"; do
    if (( i % 2 == 0 )); then A_IDX+=("$i"); else B_IDX+=("$i"); fi
done

echo "[$(date -Iseconds)] === T172 DUAL re-run BEGIN (${#CELLS[@]} cells, 2 streams) ==="
echo "  stream A gpu0:8195 idx: ${A_IDX[*]}"
echo "  stream B gpu1:8196 idx: ${B_IDX[*]}"
echo

run_stream 0 8195 "${A_IDX[@]}" &
PA=$!
run_stream 1 8196 "${B_IDX[@]}" &
PB=$!
wait $PA
wait $PB

echo
echo "[$(date -Iseconds)] === T172 DUAL re-run DONE ==="
column -t -s $'\t' "$SUMMARY"
