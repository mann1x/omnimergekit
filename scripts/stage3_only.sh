#!/bin/bash
# Standalone Stage 3 wrapper (v2 — patched for PATH + partial-N + zombie kill)
# Stage 3 of post_track5_full_matrix.sh, decoupled, hardened.
set -uo pipefail

BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
RES_21Q=$BM/eval_results_corpora_21q
RES_FULL=$BM/eval_results_tracks_2_3
TS=$(date +%Y%m%d_%H%M%S)
LOG=$BM/logs/stage3_only_$TS.log

# PATCH 1: export PATH so lm-eval is visible
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1

declare -A EXPECTED_N=( [humanevalplus_full]=164 [ifeval_100]=100 [multipl_e_100]=300 )

echo "[$(date -Iseconds)] === Stage 3 standalone wrapper v2 ==="
echo "  PY:       $PY"
echo "  OMK:      $OMK"
echo "  PATH OK:  $(which lm-eval 2>/dev/null || echo MISSING)"
echo "  RES_21Q:  $RES_21Q"
echo "  RES_FULL: $RES_FULL"
echo

# Compute actual sample count per (bench, variant)
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
            # PATCH 2: purge if score=null OR n < expected
            if [ -z "$score" ] || [ "$score" = "null" ]; then
                echo "  [retry] $tpl (score=null)"
                rm -rf "$sd"
            elif [ "$n" -lt "$expected" ]; then
                echo "  [retry] $tpl (partial n=$n/$expected)"
                rm -rf "$sd"
            else
                echo "  [skip]  $tpl (n=$n/$expected score=$score)"
                continue
            fi
        fi
        # PATCH 3: timeout per bench to escape zombie-llama hangs
        # 30 min cap for HE+/IF (~10 min typical), 60 min for MPE (~40 min typical)
        local tlim=1800
        [ "$tpl" = "multipl_e_100" ] && tlim=3600
        timeout --kill-after=10 --signal=KILL $tlim \
            "$PY" "$OMK" --model "$q6" --tokenizer "$tok" --template $tpl --backend llama \
            --served-name "$served" --results-dir "$RES_FULL" 2>&1 | tail -10
        rc=${PIPESTATUS[0]}
        echo "  [done]  $tpl rc=$rc"
        # Mop up any stale llama-server on the matrix port
        if pgrep -f "llama-server.*--port 8195" >/dev/null; then
            echo "  [cleanup] killing leftover llama-server on 8195"
            pkill -KILL -f "llama-server.*--port 8195"
            sleep 2
        fi
    done
}

for label in calibonly 9bench ifheavy; do
    for method in eac rkd kdonly; do
        served="a2${method}_${label}-62e-fc15_25-p8-s1_0p1_20"
        if_score=$(jq -r '.score // 0' "$RES_21Q/ifeval_rum3/$served/summary.json" 2>/dev/null || echo 0)
        he_score=$(jq -r '.score // 0' "$RES_21Q/humanevalplus_rum3/$served/summary.json" 2>/dev/null || echo 0)
        mp_score=$(jq -r '.score // 0' "$RES_21Q/multipl_e_rum15/$served/summary.json" 2>/dev/null || echo 0)

        run_full=0
        awk "BEGIN{exit ($if_score >= 0.333) ? 0 : 1}" && run_full=1
        awk "BEGIN{exit ($he_score >= 0.333) ? 0 : 1}" && run_full=1
        awk "BEGIN{exit ($mp_score > 0.20) ? 0 : 1}" && run_full=1

        if [ $run_full -eq 1 ]; then
            out_dir=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-${method}-${label}-it
            q6=${out_dir}-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-${method}-${label}-it-Q6_K.gguf
            tok_dir="${out_dir}-GGUF"
            if [ ! -f "$q6" ]; then
                echo "[$(date -Iseconds)] [missing q6] $served — SKIP"
                continue
            fi
            if [ ! -f "$tok_dir/tokenizer.json" ]; then
                echo "[$(date -Iseconds)] [missing tok] $tok_dir — SKIP"
                continue
            fi
            echo
            echo "[$(date -Iseconds)] >>> full bench $served (HE+=$he_score IFEval=$if_score MPE=$mp_score)"
            bench_full "$q6" "$tok_dir" "$served"
        else
            echo "[$(date -Iseconds)] [gate skip] $served"
        fi
    done
done

echo
echo "[$(date -Iseconds)] === Stage 3 standalone v2 complete ==="
