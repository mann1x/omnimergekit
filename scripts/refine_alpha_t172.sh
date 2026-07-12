#!/bin/bash
# refine_alpha_t172.sh — T172 Phase 1 bisection refinement.
#
# Read phase1_coarse_<knob>_summary.tsv, pick winner from coarse {1.10, 1.20,
# 1.30} (or whatever was swept), probe ±0.05 neighbors. If a neighbor beats
# the winner, continue 0.05 steps in that direction until any flag triggers
# OR two consecutive probes degrade. Append every probe to
# phase1_refine_<knob>_summary.tsv.
#
# Winner criterion (apples-to-apples with user's bisection instruction):
#   CLEAN_ALL preferred. Tiebreak by max(HE+) then max(MPE) then min(|IF Δ|).
#
# Calls the same build_alpha_variant.sh + bench_full + emit_audit_row as the
# coarse sweep. Idempotent on per-α .audit_ready markers.
#
# Usage:
#   refine_alpha_t172.sh --knob shared
#   refine_alpha_t172.sh --knob pes
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

KNOB=""
MAX_PROBES=6  # safety cap on the 0.05-step chain
while [ $# -gt 0 ]; do
    case "$1" in
        --knob)      KNOB=$2; shift 2 ;;
        --max-probes) MAX_PROBES=$2; shift 2 ;;
        *)           echo "FATAL: unknown $1"; exit 2 ;;
    esac
done
[ "$KNOB" = "shared" ] || [ "$KNOB" = "pes" ] || { echo "FATAL: --knob {shared|pes}"; exit 2; }

COARSE_TSV=$LOG_DIR/phase1_coarse_${KNOB}_summary.tsv
REFINE_TSV=$LOG_DIR/phase1_refine_${KNOB}_summary.tsv
[ -s "$COARSE_TSV" ] || { echo "FATAL: $COARSE_TSV missing — run sweep_alpha_t172.sh first"; exit 2; }

TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/refine_${KNOB}_${TS}.log
exec > >(tee "$LOG") 2>&1

declare -A EXPECTED_N=( [humanevalplus_full]=164 [ifeval_100]=100 [multipl_e_100]=300 )

echo "[$(date -Iseconds)] === T172 Phase 1 refine — knob=$KNOB ==="
echo "  coarse TSV: $COARSE_TSV"
echo "  refine TSV: $REFINE_TSV"
echo "  max probes: $MAX_PROBES"
echo

# Build TSV header if new
if [ ! -s "$REFINE_TSV" ]; then
    printf 'ts\tknob\talpha\tvariant\the_score\the_delta\the_flags\tif_score\tif_delta\tif_flags\tmpe_score\tmpe_delta\tmpe_flags\tverdict\n' > "$REFINE_TSV"
fi

# --- helpers (lifted from sweep + stage3) ---
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
            else
                continue
            fi
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
variant_name() {
    local knob=$1 alpha=$2
    local int=$(awk -v a="$alpha" 'BEGIN{ printf("%d", a*100 + 0.5) }')
    echo "gemma-4-A4B-62e-fc15_25-p8-${knob}${int}-it"
}
emit_audit_row() {
    local alpha=$1 variant=$2 tsv=$3
    local he_line if_line mp_line
    he_line=$("$PY" "$AUDIT" humanevalplus_full "$variant" "$BASELINE" 2>/dev/null | grep '^AUDIT' || echo "AUDIT_FAIL  humanevalplus_full")
    if_line=$("$PY" "$AUDIT" ifeval_100         "$variant" "$BASELINE" 2>/dev/null | grep '^AUDIT' || echo "AUDIT_FAIL  ifeval_100")
    mp_line=$("$PY" "$AUDIT" multipl_e_100      "$variant" "$BASELINE" 2>/dev/null | grep '^AUDIT' || echo "AUDIT_FAIL  multipl_e_100")
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
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Iseconds)" "$KNOB" "$alpha" "$variant" \
        "$he_s" "$he_d" "$he_f" "$if_s" "$if_d" "$if_f" \
        "$mp_s" "$mp_d" "$mp_f" "$verdict" >> "$tsv"
    echo "    >>> TSV row: alpha=$alpha verdict=$verdict"
    # Return values via globals for caller
    LAST_VERDICT=$verdict
    LAST_HE=$he_s; LAST_IF=$if_s; LAST_MP=$mp_s
}

# --- pick winner from coarse (CLEAN_ALL preferred; tiebreak HE+ > MPE > |IF Δ|) ---
pick_winner() {
    "$PY" - "$COARSE_TSV" "$KNOB" <<'PYEOF'
import sys, csv
tsv, knob = sys.argv[1], sys.argv[2]
rows = [r for r in csv.DictReader(open(tsv), delimiter='\t') if r['knob'] == knob]
def score(r):
    clean = 1 if r['verdict']=='CLEAN_ALL' else 0
    try:
        he = float(r['he_score']); mp = float(r['mpe_score']); ifd = abs(float(r['if_delta']))
    except: he = mp = 0.0; ifd = 9.0
    return (clean, he, mp, -ifd)
if not rows:
    print("0.00"); sys.exit(0)
best = max(rows, key=score)
print(best['alpha'])
PYEOF
}

WINNER_A=$(pick_winner)
echo "[$(date -Iseconds)] coarse winner for knob=$KNOB: α=$WINNER_A"

# --- bisection chain: probe ±0.05, continue in winning direction ---
already_done() {
    local a=$1
    grep -P "^[^\t]+\t${KNOB}\t${a}\t" "$COARSE_TSV" "$REFINE_TSV" 2>/dev/null | head -1
}
get_score_from_tsv() {
    # echo (verdict he_score) for the alpha
    local a=$1
    grep -P "^[^\t]+\t${KNOB}\t${a}\t" "$COARSE_TSV" "$REFINE_TSV" 2>/dev/null | head -1 | awk -F'\t' '{print $14, $5}'
}
build_and_audit() {
    local a=$1
    local variant=$(variant_name "$KNOB" "$a")
    local out=$BM_WORKING/google/$variant
    local q6=${out}-GGUF/${variant}-Q6_K.gguf
    local tok_dir="${out}-GGUF"

    if [ ! -f "$q6" ]; then
        echo "  [$(date -Iseconds)] build α=$a"
        case "$KNOB" in
            shared) "$BUILDER" --src "$PRISTINE" --shared-alpha "$a" --pes-alpha 1.00 --order shared-first --out "$out" 2>&1 | tail -15 ;;
            pes)    "$BUILDER" --src "$PRISTINE" --shared-alpha 1.00 --pes-alpha "$a" --order pes-first    --out "$out" 2>&1 | tail -15 ;;
        esac
        [ -f "$q6" ] || { echo "  FATAL: build failed α=$a"; return 1; }
    fi
    bench_full "$q6" "$tok_dir" "$variant"
    emit_audit_row "$a" "$variant" "$REFINE_TSV"
}

# Compare verdict (higher score wins). Returns 0 if a beats winner, 1 if not.
better_than_winner() {
    local cand_he=$1 cand_v=$2
    local win_v=$3 win_he=$4
    # CLEAN_ALL beats non-CLEAN_ALL
    if [ "$cand_v" = "CLEAN_ALL" ] && [ "$win_v" != "CLEAN_ALL" ]; then return 0; fi
    if [ "$cand_v" != "CLEAN_ALL" ] && [ "$win_v" = "CLEAN_ALL" ]; then return 1; fi
    # Same verdict class — compare HE+
    awk "BEGIN{exit ($cand_he > $win_he) ? 0 : 1}"
}

direction=""    # "up" or "down" or "" (undetermined)
current_a=$WINNER_A
current_verdict=$(get_score_from_tsv "$current_a" | awk '{print $1}')
current_he=$(get_score_from_tsv "$current_a" | awk '{print $2}')
echo "  current winner: α=$current_a verdict=$current_verdict he=$current_he"

probe_count=0
degrade_count=0

# Probe both ±0.05 neighbors first to determine direction
for neighbor_op in down up; do
    case "$neighbor_op" in
        down) probe_a=$(awk "BEGIN{ printf(\"%.2f\", $current_a - 0.05) }") ;;
        up)   probe_a=$(awk "BEGIN{ printf(\"%.2f\", $current_a + 0.05) }") ;;
    esac
    awk "BEGIN{exit ($probe_a > 0) ? 0 : 1}" || { echo "  skip $neighbor_op probe ($probe_a ≤ 0)"; continue; }
    if existing=$(already_done "$probe_a"); then
        echo "  [$(date -Iseconds)] neighbor α=$probe_a already done — reading TSV"
    else
        probe_count=$((probe_count + 1))
        [ $probe_count -gt "$MAX_PROBES" ] && { echo "  hit MAX_PROBES=$MAX_PROBES — stopping"; break; }
        echo "[$(date -Iseconds)] >>> neighbor probe ($neighbor_op): α=$probe_a"
        build_and_audit "$probe_a" || continue
    fi
    LV=$(get_score_from_tsv "$probe_a" | awk '{print $1}')
    LH=$(get_score_from_tsv "$probe_a" | awk '{print $2}')
    if better_than_winner "$LH" "$LV" "$current_verdict" "$current_he"; then
        echo "  α=$probe_a (he=$LH verdict=$LV) BEATS current α=$current_a (he=$current_he verdict=$current_verdict)"
        direction=$neighbor_op
        current_a=$probe_a
        current_verdict=$LV
        current_he=$LH
        break
    else
        echo "  α=$probe_a does NOT beat α=$current_a"
    fi
done

if [ -z "$direction" ]; then
    echo "[$(date -Iseconds)] Neither ±0.05 neighbor beats coarse winner α=$WINNER_A — refinement DONE"
else
    # Continue in direction until 2 consecutive degradations or flag-trigger
    while [ $probe_count -lt "$MAX_PROBES" ]; do
        case "$direction" in
            up)   probe_a=$(awk "BEGIN{ printf(\"%.2f\", $current_a + 0.05) }") ;;
            down) probe_a=$(awk "BEGIN{ printf(\"%.2f\", $current_a - 0.05) }") ;;
        esac
        awk "BEGIN{exit ($probe_a > 0) ? 0 : 1}" || { echo "  α=$probe_a ≤ 0 — stop"; break; }
        if existing=$(already_done "$probe_a"); then
            echo "  [$(date -Iseconds)] α=$probe_a already done — reading TSV"
        else
            probe_count=$((probe_count + 1))
            echo "[$(date -Iseconds)] >>> continue $direction: α=$probe_a (probe $probe_count/$MAX_PROBES)"
            build_and_audit "$probe_a" || break
        fi
        LV=$(get_score_from_tsv "$probe_a" | awk '{print $1}')
        LH=$(get_score_from_tsv "$probe_a" | awk '{print $2}')
        # Flag-trigger stop
        if [ "$LV" != "CLEAN_ALL" ]; then
            echo "  α=$probe_a triggered flag (verdict=$LV) — STOP"
            break
        fi
        # Compare to current
        if better_than_winner "$LH" "$LV" "$current_verdict" "$current_he"; then
            current_a=$probe_a
            current_verdict=$LV
            current_he=$LH
            degrade_count=0
            echo "  α=$probe_a improves — continue"
        else
            degrade_count=$((degrade_count + 1))
            echo "  α=$probe_a degrades (degrade_count=$degrade_count)"
            if [ $degrade_count -ge 2 ]; then
                echo "  2 consecutive degradations — STOP"
                break
            fi
        fi
    done
fi

echo
echo "[$(date -Iseconds)] === refine DONE for knob=$KNOB ==="
echo "  final winner α=$current_a (verdict=$current_verdict, he=$current_he)"
echo "  probes built : $probe_count"
echo
echo "=== refine TSV ==="
column -t -s $'\t' "$REFINE_TSV"
