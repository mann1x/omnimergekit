#!/bin/bash
# Wiki2 + extended-corpus follow-up matrix: 3 EAC corpora × 3 methods = 9 cells
# All 3 EAC corpora = 50/50 calib + wiki2 (balanced-window 262144 tok).
# KD bumped to ~3 epochs over the actual JSONL corpus (was undertrained at 100 steps).
# Per-cell cleanup: drop safetensors after Q6+21q lands (keeps GGUF + eval).
# Author: claude opus 4.7  2026-05-29
set -uo pipefail
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=$BM/logs/wiki2_followup_$TS
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/chain.log
exec > >(tee "$LOG") 2>&1

TEACHER=$BM/google/gemma-4-26B-A4B-it
A2=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it
DROP_MAP=$BM/scripts/v6coder_C6v3lcb_62e_fc15_25_p8_drop_map.json
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
RES_21Q=$BM/eval_results_wiki2_followup_21q
mkdir -p "$RES_21Q"

declare -A EAC_CORPUS=(
    [9b_w2]=$BM/scripts/eac_corpus_9bench_plus_wiki2.txt
    [if_w2]=$BM/scripts/eac_corpus_ifheavy_plus_wiki2.txt
    [ex_w2]=$BM/scripts/eac_corpus_wiki2_plus_calib.txt
)
declare -A KD_CORPUS=(
    [9b_w2]=$BM/scripts/router_calib_corpus_9bench_balanced.jsonl
    [if_w2]=$BM/scripts/router_calib_corpus_ifeval_heavy.jsonl
    [ex_w2]=$BM/scripts/router_calib_corpus.jsonl
)
# KD --max-samples per corpus (full corpus size for 3-epoch coverage)
declare -A KD_SAMPLES=( [9b_w2]=944 [if_w2]=1194 [ex_w2]=3635 )

quantize_q6() {
    local src=$1 q6=$2
    [ -f "$q6" ] && { echo "  [quant skip] $q6"; return 0; }
    local gguf_dir=$(dirname "$q6")
    local f16=${gguf_dir}/$(basename "$q6" -Q6_K.gguf)-F16.gguf
    mkdir -p "$gguf_dir"
    "$PY" $BM/tools/llama.cpp/convert_hf_to_gguf.py "$src" --outfile "$f16" --outtype f16 2>&1 | tail -3
    /opt/llama.cpp/build/bin/llama-quantize "$f16" "$q6" Q6_K 2>&1 | tail -3
    rm -f "$f16"
    [ -f "$q6" ] || { echo "  [quant FAIL] $q6"; return 1; }
    echo "  [quant OK] $q6 ($(du -sh "$q6" | cut -f1))"
}

apply_eac() {
    local out=$1 corpus=$2 name=$3
    [ -d "$out" ] && [ -f "$out/model-00006-of-00006.safetensors" ] && { echo "  [eac skip] $out"; return 0; }
    "$PY" $BM/scripts/router_eac_calibrate.py \
        --base-dir "$TEACHER" --variant-dir "$A2" --out-dir "$out" \
        --drop-map "$DROP_MAP" --corpus-file "$corpus" \
        --n-seq 128 --seq-len 2048 --batch-size 4 \
        --calib-k 16 --lr 1e-3 --steps 150 \
        --max-gpu-gib 80 --max-cpu-gib 400 \
        --cache-dir $BM/eac_cache_${name} 2>&1 | tail -8
}

apply_kd() {
    local src=$1 out=$2 corpus=$3 name=$4 max_samples=$5
    [ -d "$out" ] && [ -f "$out/model-00006-of-00006.safetensors" ] && { echo "  [kd skip] $out"; return 0; }
    "$PY" $BM/scripts/router_kd.py \
        --base-dir "$TEACHER" --variant-dir "$src" --out-dir "$out" \
        --teacher-load bf16 --student-load bf16 \
        --teacher-device "{\"\":0}" --student-device "{\"\":1}" \
        --tau 1.0 --lr 1e-5 --max-steps 0 \
        --batch-size 2 --grad-accum 4 \
        --max-seq-len 512 --max-samples ${max_samples} \
        --epochs 3 \
        --corpus-file "$corpus" \
        --checkpoint-dir $LOG_DIR/ckpt_${name} \
        --canary-file $BM/scripts/ifeval_rumination_canaries.json \
        --no-canary 2>&1 | tail -8
}

bench_21q() {
    local q6=$1 tok=$2 served=$3
    for tpl in humanevalplus_rum3 ifeval_rum3 multipl_e_rum15; do
        [ -f "$RES_21Q/$tpl/$served/summary.json" ] && continue
        "$PY" "$OMK" --model "$q6" --tokenizer "$tok" --template $tpl --backend llama \
            --served-name "$served" --results-dir "$RES_21Q" 2>&1 | tail -10
    done
    echo "  21q $served:"
    for tpl in humanevalplus_rum3 ifeval_rum3 multipl_e_rum15; do
        s=$(jq -r ".score // \"null\"" "$RES_21Q/$tpl/$served/summary.json" 2>/dev/null || echo MISS)
        printf "    %-22s %s\n" "$tpl" "$s"
    done
}

# Cleanup: drop safetensors variant + EAC cache after Q6+21q is in eval_results.
# Keep tokenizer files for re-bench by copying tokenizer.json + chat_template into the GGUF dir.
cleanup_variant() {
    local src=$1 q6=$2 served=$3
    [ -f "$q6" ] || { echo "  [cleanup skip] no Q6 for $served"; return 1; }
    for tpl in humanevalplus_rum3 ifeval_rum3 multipl_e_rum15; do
        [ -f "$RES_21Q/$tpl/$served/summary.json" ] || { echo "  [cleanup skip] 21q $tpl missing"; return 1; }
    done
    local gguf_dir=$(dirname "$q6")
    cp -n "$src"/tokenizer*.json "$gguf_dir/" 2>/dev/null || true
    cp -n "$src"/chat_template.jinja "$gguf_dir/" 2>/dev/null || true
    cp -n "$src"/config.json "$gguf_dir/" 2>/dev/null || true
    cp -n "$src"/generation_config.json "$gguf_dir/" 2>/dev/null || true
    local sz_before=$(du -shx "$src" 2>/dev/null | cut -f1)
    rm -rf "$src"
    echo "  [cleanup] purged $src ($sz_before)"
}

cleanup_eac_cache() {
    local name=$1
    local cache=$BM/eac_cache_${name}
    if [ -d "$cache" ]; then
        local sz=$(du -shx "$cache" 2>/dev/null | cut -f1)
        rm -rf "$cache"
        echo "  [cleanup] purged $cache ($sz)"
    fi
}

echo "[$(date -Iseconds)] === wiki2+extended follow-up matrix start ==="
echo "  KD epochs=3, full per-corpus coverage"
echo "  per-cell cleanup: safetensors purged after Q6+21q"
echo "  res 21q: $RES_21Q"
echo "  log dir: $LOG_DIR"
df -h / | tail -1
echo

# === Loop 1: EAC-only + EAC+KD per corpus ===
for label in 9b_w2 if_w2 ex_w2; do
    eac_corpus=${EAC_CORPUS[$label]}
    kd_corpus=${KD_CORPUS[$label]}
    kd_samples=${KD_SAMPLES[$label]}
    EAC_OUT=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-${label}-it
    EAC_Q6_DIR=${EAC_OUT}-GGUF
    EAC_Q6=${EAC_Q6_DIR}/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-${label}-it-Q6_K.gguf
    echo "[$(date -Iseconds)] >>> EAC-only $label  (corpus=$(basename "$eac_corpus"))"
    apply_eac "$EAC_OUT" "$eac_corpus" "eac_$label"
    quantize_q6 "$EAC_OUT" "$EAC_Q6"
    EAC_SERVED="a2eac_${label}-62e-fc15_25-p8-s1_0p1_20"
    bench_21q "$EAC_Q6" "$EAC_OUT" "$EAC_SERVED"

    RKD_OUT=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-${label}-it
    RKD_Q6_DIR=${RKD_OUT}-GGUF
    RKD_Q6=${RKD_Q6_DIR}/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-${label}-it-Q6_K.gguf
    echo "[$(date -Iseconds)] >>> EAC+KD $label  (kd_corpus=$(basename "$kd_corpus") samples=$kd_samples epochs=3)"
    apply_kd "$EAC_OUT" "$RKD_OUT" "$kd_corpus" "rkd_$label" "$kd_samples"
    quantize_q6 "$RKD_OUT" "$RKD_Q6"
    RKD_SERVED="a2rkd_${label}-62e-fc15_25-p8-s1_0p1_20"
    bench_21q "$RKD_Q6" "$RKD_OUT" "$RKD_SERVED"

    cleanup_variant "$EAC_OUT" "$EAC_Q6" "$EAC_SERVED"
    cleanup_variant "$RKD_OUT" "$RKD_Q6" "$RKD_SERVED"
    cleanup_eac_cache "eac_$label"
    df -h / | tail -1
done

# === Loop 2: KD-only on raw A2 ===
for label in 9b_w2 if_w2 ex_w2; do
    kd_corpus=${KD_CORPUS[$label]}
    kd_samples=${KD_SAMPLES[$label]}
    KDO_OUT=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-${label}-it
    KDO_Q6_DIR=${KDO_OUT}-GGUF
    KDO_Q6=${KDO_Q6_DIR}/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-${label}-it-Q6_K.gguf
    echo "[$(date -Iseconds)] >>> KD-only $label  (kd_corpus=$(basename "$kd_corpus") samples=$kd_samples epochs=3)"
    apply_kd "$A2" "$KDO_OUT" "$kd_corpus" "kdonly_$label" "$kd_samples"
    quantize_q6 "$KDO_OUT" "$KDO_Q6"
    KDO_SERVED="a2kdonly_${label}-62e-fc15_25-p8-s1_0p1_20"
    bench_21q "$KDO_Q6" "$KDO_OUT" "$KDO_SERVED"
    cleanup_variant "$KDO_OUT" "$KDO_Q6" "$KDO_SERVED"
    df -h / | tail -1
done

echo
echo "[$(date -Iseconds)] === wiki2 follow-up matrix complete ==="
echo "  results: $RES_21Q"
echo "  per-cell 21q summary table:"
for label in 9b_w2 if_w2 ex_w2; do
    for kind in a2eac a2rkd a2kdonly; do
        served="${kind}_${label}-62e-fc15_25-p8-s1_0p1_20"
        printf "  %-40s " "$served"
        for tpl in humanevalplus_rum3 ifeval_rum3 multipl_e_rum15; do
            s=$(jq -r ".score // \"null\"" "$RES_21Q/$tpl/$served/summary.json" 2>/dev/null || echo MISS)
            printf "%-6s " "${s:0:5}"
        done
        echo
    done
done
