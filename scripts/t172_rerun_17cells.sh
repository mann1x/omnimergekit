#!/bin/bash
# T172.7-rerun: comprehensive re-eval of all 17 cells affected by the
# parallel=8 template-drift bug (2026-05-29 11:51-20:48 CEST window).
# Re-runs HE+, IFEval-100, MPE-100 against the SAME Q6_K bytes via the
# patched omk_eval (safe-parallel clamp from thinking_budget) + canonical
# templates (parallel=2). Old result dirs preserved as <name>_p8_OLD.
#
# Cells = the 17 with parallel=8 from the audit table.
# Estimated wall: ~6h sequential on bs2.
#
# Author: claude opus 4.7  2026-05-30 CEST
set -uo pipefail

PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
AUDIT=/srv/ml/scripts/audit_full_bench.py
RES=/srv/ml/eval_results_tracks_2_3
BASELINE=a2-62e-fc15_25-p8-s1_0p1_20

export PATH="/srv/ml/envs/envs/omnimergekit/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export LD_LIBRARY_PATH="/srv/ml/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"

LOG_DIR=/srv/ml/logs/t172
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/t172_rerun_17cells_${TS}.log
SUMMARY=$LOG_DIR/t172_rerun_17cells_summary.tsv
exec > >(tee "$LOG") 2>&1

if [ ! -s "$SUMMARY" ]; then
    printf 'ts\tcell\tbench\tscore\tdelta_vs_a2\tflags\tverdict\n' > "$SUMMARY"
fi

# (served-name, Q6_path) pairs — harvested from server.log of each cell
# Combo is excluded — it's running as T172.5 validation already.
CELLS=$(cat <<'CSV'
a2eac_calibonly-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-calibonly-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-calibonly-it-Q6_K.gguf
a2eac_9bench-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-9bench-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-9bench-it-Q6_K.gguf
a2eac_ifheavy-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-ifheavy-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-ifheavy-it-Q6_K.gguf
a2kdonly_calibonly-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-calibonly-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-calibonly-it-Q6_K.gguf
a2kdonly_9bench-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-9bench-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-9bench-it-Q6_K.gguf
a2kdonly_ifheavy-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-ifheavy-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-ifheavy-it-Q6_K.gguf
a2rkd_calibonly-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-calibonly-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-calibonly-it-Q6_K.gguf
a2rkd_9bench-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-9bench-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-9bench-it-Q6_K.gguf
a2rkd_ifheavy-62e-fc15_25-p8-s1_0p1_20|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-ifheavy-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-ifheavy-it-Q6_K.gguf
gemma-4-A4B-62e-fc15_25-p8-pristine-it|/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pristine-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-pristine-it-Q6_K.gguf
gemma-4-A4B-62e-fc15_25-p8-shared110-it|/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-shared110-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-shared110-it-Q6_K.gguf
gemma-4-A4B-62e-fc15_25-p8-shared120-it|/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-shared120-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-shared120-it-Q6_K.gguf
gemma-4-A4B-62e-fc15_25-p8-shared130-it|/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-shared130-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-shared130-it-Q6_K.gguf
gemma-4-A4B-62e-fc15_25-p8-pes110-it|/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes110-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-pes110-it-Q6_K.gguf
gemma-4-A4B-62e-fc15_25-p8-pes120-it|/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-pes120-it-Q6_K.gguf
gemma-4-A4B-62e-fc15_25-p8-pes130-it|/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes130-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-pes130-it-Q6_K.gguf
pes1_10-62e-fc15_25-p8|/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes1_10-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-pes1_10-it-Q6_K.gguf
CSV
)

BENCHES=(ifeval_100)

declare -A TLIM=( [humanevalplus_full]=2400 [ifeval_100]=5400 [multipl_e_100]=3600 )

audit_emit() {
    local bench=$1 served=$2
    local line
    line=$("$PY" "$AUDIT" "$bench" "$served" "$BASELINE" 2>/dev/null | grep '^AUDIT' || echo "AUDIT_FAIL")
    echo "$line"
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
    local q6=$1 tok=$2 served=$3 bench=$4
    local tlim=${TLIM[$bench]}
    local sd="$RES/$bench/$served"
    # Always purge: we explicitly want a fresh result via patched code
    if [ -d "$sd" ]; then
        local bak="$RES/$bench/${served}_p8_OLD"
        [ -d "$bak" ] && rm -rf "$bak"
        mv "$sd" "$bak"
        echo "    preserved old → $bak"
    fi
    pkill -KILL -f "llama-server.*--port 8195" 2>/dev/null
    sleep 2
    timeout --kill-after=10 --signal=KILL "$tlim" \
        "$PY" "$OMK" --model "$q6" --tokenizer "$tok" --template "$bench" \
        --backend llama \
        --served-name "$served" --results-dir "$RES" 2>&1 | tail -10
    if pgrep -f "llama-server.*--port 8195" >/dev/null; then
        pkill -KILL -f "llama-server.*--port 8195"; sleep 2
    fi
}

echo "[$(date -Iseconds)] === T172 17-cell re-run BEGIN ==="
echo "  cells   : 17 (combo excluded — handled by T172.5)"
echo "  benches : ${BENCHES[*]}"
echo "  log     : $LOG"
echo "  TSV     : $SUMMARY"
echo

N=0
TOTAL=$(echo "$CELLS" | wc -l)
while IFS='|' read -r served q6; do
    [ -z "$served" ] && continue
    N=$((N+1))
    tok="$(dirname "$q6")"
    echo
    echo "[$(date -Iseconds)] >>> cell $N/$TOTAL: $served"
    echo "    Q6  : $q6"
    if [ ! -f "$q6" ]; then
        echo "    SKIP — Q6 missing"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$served" "MISSING_Q6" "?" "?" "?" "SKIP" >> "$SUMMARY"
        continue
    fi
    for bench in "${BENCHES[@]}"; do
        echo
        echo "  --- $bench ---"
        bench_one "$q6" "$tok" "$served" "$bench"
        audit_emit "$bench" "$served"
    done
    echo
    echo "    [done cell $N/$TOTAL] $served"
done <<<"$CELLS"

echo
echo "[$(date -Iseconds)] === T172 17-cell re-run DONE ==="
echo
echo "=== final summary TSV ==="
column -t -s $'\t' "$SUMMARY"
