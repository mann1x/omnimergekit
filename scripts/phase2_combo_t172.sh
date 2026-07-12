#!/bin/bash
# phase2_combo_t172.sh — T172 Phase 2 combination sweep.
#
# Given (shared α, pes α) winners from Phase 1, run BOTH orderings:
#   - shared-first: cp pristine -> apply shared α -> apply pes α
#   - pes-first:    cp pristine -> apply pes α    -> apply shared α
# Append both audit results to phase2_combo_summary.tsv.
#
# Gated on Phase 1 outcome A or B (per plan §"Phase 1 GATE"). User confirms
# winners before launch.
#
# Usage:
#   phase2_combo_t172.sh --shared-alpha 1.15 --pes-alpha 1.20
#
# Author: claude opus 4.7  2026-05-29
set -uo pipefail

BM=/srv/ml
BM_WORKING=${BM_WORKING:-/mnt/sdc/ml}
PY=$BM/envs/envs/omnimergekit/bin/python
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
AUDIT=$BM/scripts/audit_full_bench.py
BUILDER=$BM/scripts/build_alpha_variant.sh

PRISTINE=$BM_WORKING/google/gemma-4-A4B-62e-fc15_25-p8-pristine-it
RES_FULL=$BM/eval_results_tracks_2_3
BASELINE=a2-62e-fc15_25-p8-s1_0p1_20

LOG_DIR=$BM/logs/t172
mkdir -p "$LOG_DIR"

export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"

SHARED_A=""; PES_A=""
while [ $# -gt 0 ]; do
    case "$1" in
        --shared-alpha) SHARED_A=$2; shift 2 ;;
        --pes-alpha)    PES_A=$2; shift 2 ;;
        *)              echo "FATAL: unknown $1"; exit 2 ;;
    esac
done
[ -n "$SHARED_A" ] && [ -n "$PES_A" ] || { echo "FATAL: --shared-alpha and --pes-alpha required"; exit 2; }

SUMMARY=$LOG_DIR/phase2_combo_summary.tsv
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/phase2_combo_${TS}.log
exec > >(tee "$LOG") 2>&1

declare -A EXPECTED_N=( [humanevalplus_full]=164 [ifeval_100]=100 [multipl_e_100]=300 )

echo "[$(date -Iseconds)] === T172 Phase 2 combo ==="
echo "  shared α : $SHARED_A"
echo "  pes α    : $PES_A"
echo

if [ ! -s "$SUMMARY" ]; then
    printf 'ts\torder\tshared_alpha\tpes_alpha\tvariant\the_score\the_delta\the_flags\tif_score\tif_delta\tif_flags\tmpe_score\tmpe_delta\tmpe_flags\tverdict\n' > "$SUMMARY"
fi

samples_n() {
    local bench=$1 variant=$2
    local sd="$RES_FULL/$bench/$variant"
    if [ "$bench" = "multipl_e_100" ]; then
        [ -f "$sd/mpe_result.samples.jsonl" ] && wc -l < "$sd/mpe_result.samples.jsonl" || echo 0
    else
        local f
        f=$(find "$sd/lm_eval_out" -maxdepth 3 -name "samples_*.jsonl" 2>/dev/null | head -1)
        [ -n "$f" ] && wc -l < "$f" || echo 0
    fi
}
bench_full() {
    local q6=$1 tok=$2 served=$3
    for tpl in humanevalplus_full ifeval_100 multipl_e_100; do
        local expected=${EXPECTED_N[$tpl]}
        local sd="$RES_FULL/$tpl/$served"
        if [ -f "$sd/summary.json" ]; then
            local score n
            score=$(jq -r '.score // empty' "$sd/summary.json" 2>/dev/null)
            n=$(samples_n "$tpl" "$served")
            if [ -z "$score" ] || [ "$score" = "null" ] || [ "$n" -lt "$expected" ]; then
                rm -rf "$sd"
            else continue; fi
        fi
        local tlim=1800
        [ "$tpl" = "multipl_e_100" ] && tlim=3600
        timeout --kill-after=10 --signal=KILL $tlim \
            "$PY" "$OMK" --model "$q6" --tokenizer "$tok" --template $tpl --backend llama \
            --served-name "$served" --results-dir "$RES_FULL" 2>&1 | tail -10
        if pgrep -f "llama-server.*--port 8195" >/dev/null; then
            pkill -KILL -f "llama-server.*--port 8195"; sleep 2
        fi
    done
}

# Variant naming: combo cells get explicit -shared{a}-pes{b}-{order}-it suffix
combo_variant_name() {
    local sa=$1 pa=$2 ord=$3
    local sint=$(awk -v a="$sa" 'BEGIN{ printf("%d", a*100 + 0.5) }')
    local pint=$(awk -v a="$pa" 'BEGIN{ printf("%d", a*100 + 0.5) }')
    local osuf=""
    case "$ord" in shared-first) osuf="sf" ;; pes-first) osuf="pf" ;; esac
    echo "gemma-4-A4B-62e-fc15_25-p8-shared${sint}-pes${pint}-${osuf}-it"
}

emit_combo_row() {
    local order=$1 sa=$2 pa=$3 variant=$4
    local he_line if_line mp_line
    he_line=$("$PY" "$AUDIT" humanevalplus_full "$variant" "$BASELINE" 2>/dev/null | grep '^AUDIT' || echo "AUDIT_FAIL")
    if_line=$("$PY" "$AUDIT" ifeval_100         "$variant" "$BASELINE" 2>/dev/null | grep '^AUDIT' || echo "AUDIT_FAIL")
    mp_line=$("$PY" "$AUDIT" multipl_e_100      "$variant" "$BASELINE" 2>/dev/null | grep '^AUDIT' || echo "AUDIT_FAIL")
    echo "$he_line"; echo "$if_line"; echo "$mp_line"
    pf() { echo "$1" | sed -nE "s/.*\b${2}=([^[:space:]]+).*/\1/p" | head -1; }
    pflags() { echo "$1" | sed -nE 's/.*flags=\[([^]]*)\].*/\1/p' | head -1; }
    local he_s he_d he_f if_s if_d if_f mp_s mp_d mp_f
    he_s=$(pf "$he_line" score); he_d=$(pf "$he_line" d); he_f=$(pflags "$he_line")
    if_s=$(pf "$if_line" score); if_d=$(pf "$if_line" d); if_f=$(pflags "$if_line")
    mp_s=$(pf "$mp_line" score); mp_d=$(pf "$mp_line" d); mp_f=$(pflags "$mp_line")
    local verdict="MIXED"
    if [[ "$he_f" == "CLEAN" && "$if_f" == "CLEAN" && "$mp_f" == "CLEAN" ]]; then verdict="CLEAN_ALL"
    elif [[ "$he_f$if_f$mp_f" == *"PARTIAL_BENCH"* ]]; then verdict="PARTIAL"
    elif [[ "$he_f$if_f$mp_f" == *"SAT_COLLAPSE"* ]]; then verdict="SAT_COLLAPSE"
    elif [[ "$he_f$if_f$mp_f" == *"KNOWLEDGE_SHIFT"* ]]; then verdict="KNOWLEDGE_SHIFT"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Iseconds)" "$order" "$sa" "$pa" "$variant" \
        "$he_s" "$he_d" "$he_f" "$if_s" "$if_d" "$if_f" \
        "$mp_s" "$mp_d" "$mp_f" "$verdict" >> "$SUMMARY"
    echo "    >>> TSV row: order=$order verdict=$verdict"
}

for order in shared-first pes-first; do
    variant=$(combo_variant_name "$SHARED_A" "$PES_A" "$order")
    out=$BM_WORKING/google/$variant
    q6=${out}-GGUF/${variant}-Q6_K.gguf
    tok_dir="${out}-GGUF"

    echo
    echo "[$(date -Iseconds)] >>> combo cell: shared=$SHARED_A pes=$PES_A order=$order"

    if [ ! -f "$q6" ]; then
        "$BUILDER" --src "$PRISTINE" \
            --shared-alpha "$SHARED_A" --pes-alpha "$PES_A" \
            --order "$order" --out "$out" 2>&1 | tail -20
        [ -f "$q6" ] || { echo "    FATAL: build failed — skipping audit"; continue; }
    fi
    bench_full "$q6" "$tok_dir" "$variant"
    emit_combo_row "$order" "$SHARED_A" "$PES_A" "$variant"
done

echo
echo "[$(date -Iseconds)] === Phase 2 combo DONE ==="
echo
echo "=== combo TSV ==="
column -t -s $'\t' "$SUMMARY"
