#!/bin/bash
# sweep_alpha_t172.sh — T172 Phase 1 coarse α-sweep orchestrator.
#
# For each α in --alphas list, build the variant (idempotent), run full benches
# (HE+_full, IFEval-100, MPE-100), then call audit_full_bench.py per bench.
# Emits TSV summary per cell.
#
# Reuses the bench_full() pattern from stage3_only.sh (PATH export, partial-N
# retry, per-bench timeout, port-8195 mop-up) so resilience matches the matrix.
#
# Usage:
#   sweep_alpha_t172.sh --knob shared --alphas 1.00,1.10,1.20,1.30
#   sweep_alpha_t172.sh --knob pes    --alphas 1.00,1.10,1.20,1.30
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

# --- PATH for lm-eval (matches stage3_only.sh) ---
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"

# --- parse args ---
KNOB=""; ALPHAS="1.00,1.10,1.20,1.30"
while [ $# -gt 0 ]; do
    case "$1" in
        --knob)   KNOB=$2; shift 2 ;;
        --alphas) ALPHAS=$2; shift 2 ;;
        *)        echo "FATAL: unknown arg $1"; exit 2 ;;
    esac
done
[ "$KNOB" = "shared" ] || [ "$KNOB" = "pes" ] || { echo "FATAL: --knob {shared|pes}"; exit 2; }
[ -f "$AUDIT" ]        || { echo "FATAL: audit script missing at $AUDIT"; exit 2; }
[ -f "$BUILDER" ]      || { echo "FATAL: builder missing at $BUILDER"; exit 2; }
[ -d "$PRISTINE" ]     || { echo "FATAL: pristine missing at $PRISTINE — run build_pristine_62e.sh first"; exit 2; }

TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/sweep_${KNOB}_${TS}.log
SUMMARY=$LOG_DIR/phase1_coarse_${KNOB}_summary.tsv
exec > >(tee "$LOG") 2>&1

declare -A EXPECTED_N=( [humanevalplus_full]=164 [ifeval_100]=100 [multipl_e_100]=300 )

echo "[$(date -Iseconds)] === T172 Phase 1 coarse sweep ==="
echo "  knob       : $KNOB"
echo "  alphas     : $ALPHAS"
echo "  pristine   : $PRISTINE"
echo "  baseline   : $BASELINE (for audit comparison)"
echo "  res-dir    : $RES_FULL"
echo "  log        : $LOG"
echo "  summary    : $SUMMARY"
echo "  PATH OK    : $(which lm-eval 2>/dev/null || echo MISSING)"
echo

# Header (only if new)
if [ ! -s "$SUMMARY" ]; then
    printf 'ts\tknob\talpha\tvariant\the_score\the_delta\the_flags\tif_score\tif_delta\tif_flags\tmpe_score\tmpe_delta\tmpe_flags\tverdict\n' > "$SUMMARY"
fi

# --- helpers lifted from stage3_only.sh ---
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
            if [ -z "$score" ] || [ "$score" = "null" ]; then
                echo "    [retry] $tpl (score=null)"
                rm -rf "$sd"
            elif [ "$n" -lt "$expected" ]; then
                echo "    [retry] $tpl (partial n=$n/$expected)"
                rm -rf "$sd"
            else
                echo "    [skip]  $tpl (n=$n/$expected score=$score)"
                continue
            fi
        fi
        local tlim=1800
        [ "$tpl" = "multipl_e_100" ] && tlim=3600
        timeout --kill-after=10 --signal=KILL $tlim \
            "$PY" "$OMK" --model "$q6" --tokenizer "$tok" --template $tpl --backend llama \
            --served-name "$served" --results-dir "$RES_FULL" 2>&1 | tail -10
        local rc=${PIPESTATUS[0]}
        echo "    [done]  $tpl rc=$rc"
        if pgrep -f "llama-server.*--port 8195" >/dev/null; then
            echo "    [cleanup] killing leftover llama-server on 8195"
            pkill -KILL -f "llama-server.*--port 8195"
            sleep 2
        fi
    done
}

# --- compose variant name from α (e.g. shared110, pes110, shared100) ---
variant_name() {
    local knob=$1 alpha=$2
    # 1.00 -> 100, 1.05 -> 105, 1.10 -> 110, 1.20 -> 120
    local int=$(awk -v a="$alpha" 'BEGIN{ printf("%d", a*100 + 0.5) }')
    echo "gemma-4-A4B-62e-fc15_25-p8-${knob}${int}-it"
}

# --- emit AUDIT line + parse for TSV row ---
emit_audit_row() {
    local knob=$1 alpha=$2 variant=$3
    local he_line if_line mp_line
    he_line=$("$PY" "$AUDIT" humanevalplus_full "$variant" "$BASELINE" 2>/dev/null | grep '^AUDIT' || echo "AUDIT_FAIL  humanevalplus_full")
    if_line=$("$PY" "$AUDIT" ifeval_100         "$variant" "$BASELINE" 2>/dev/null | grep '^AUDIT' || echo "AUDIT_FAIL  ifeval_100")
    mp_line=$("$PY" "$AUDIT" multipl_e_100      "$variant" "$BASELINE" 2>/dev/null | grep '^AUDIT' || echo "AUDIT_FAIL  multipl_e_100")

    echo "$he_line"
    echo "$if_line"
    echo "$mp_line"

    parse_field() {
        local line=$1 key=$2
        echo "$line" | sed -nE "s/.*\b${key}=([^[:space:]]+).*/\1/p" | head -1
    }
    parse_flags() {
        local line=$1
        echo "$line" | sed -nE 's/.*flags=\[([^]]*)\].*/\1/p' | head -1
    }

    local he_s he_d he_f if_s if_d if_f mp_s mp_d mp_f
    he_s=$(parse_field "$he_line" score); he_d=$(parse_field "$he_line" d); he_f=$(parse_flags "$he_line")
    if_s=$(parse_field "$if_line" score); if_d=$(parse_field "$if_line" d); if_f=$(parse_flags "$if_line")
    mp_s=$(parse_field "$mp_line" score); mp_d=$(parse_field "$mp_line" d); mp_f=$(parse_flags "$mp_line")

    # Verdict: CLEAN/CLEAN/CLEAN → CLEAN_ALL; any SAT_COLLAPSE → SAT; any PARTIAL → PARTIAL; KNOWLEDGE_SHIFT → KS
    local verdict="MIXED"
    if [[ "$he_f" == "CLEAN" && "$if_f" == "CLEAN" && "$mp_f" == "CLEAN" ]]; then
        verdict="CLEAN_ALL"
    elif [[ "$he_f$if_f$mp_f" == *"PARTIAL_BENCH"* ]]; then
        verdict="PARTIAL"
    elif [[ "$he_f$if_f$mp_f" == *"SAT_COLLAPSE"* ]]; then
        verdict="SAT_COLLAPSE"
    elif [[ "$he_f$if_f$mp_f" == *"KNOWLEDGE_SHIFT"* ]]; then
        verdict="KNOWLEDGE_SHIFT"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Iseconds)" "$knob" "$alpha" "$variant" \
        "$he_s" "$he_d" "$he_f" "$if_s" "$if_d" "$if_f" \
        "$mp_s" "$mp_d" "$mp_f" "$verdict" >> "$SUMMARY"

    echo "    >>> TSV row appended: verdict=$verdict"
}

# --- main loop over α list ---
IFS=',' read -ra ALIST <<<"$ALPHAS"
for a in "${ALIST[@]}"; do
    variant=$(variant_name "$KNOB" "$a")
    out=$BM_WORKING/google/$variant
    q6=${out}-GGUF/${variant}-Q6_K.gguf
    tok_dir="${out}-GGUF"

    echo
    echo "[$(date -Iseconds)] >>> sweep cell: knob=$KNOB α=$a variant=$variant"

    if [ ! -f "$q6" ]; then
        case "$KNOB" in
            shared) "$BUILDER" --src "$PRISTINE" --shared-alpha "$a" --pes-alpha 1.00 --order shared-first --out "$out" 2>&1 | tail -20 ;;
            pes)    "$BUILDER" --src "$PRISTINE" --shared-alpha 1.00 --pes-alpha "$a" --order pes-first    --out "$out" 2>&1 | tail -20 ;;
        esac
        if [ ! -f "$q6" ]; then
            echo "    FATAL: build failed for $variant — skipping audit"
            continue
        fi
    else
        echo "    [skip-build] $q6 already present"
    fi

    bench_full "$q6" "$tok_dir" "$variant"
    emit_audit_row "$KNOB" "$a" "$variant"
done

echo
echo "[$(date -Iseconds)] === Phase 1 coarse sweep DONE for knob=$KNOB ==="
echo
echo "=== summary TSV ==="
column -t -s $'\t' "$SUMMARY"
