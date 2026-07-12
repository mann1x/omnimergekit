#!/bin/bash
# Post-Track5 chain: COMPLETE 9-combination corpus×method matrix
# Author: claude opus 4.7  2026-05-29
# 3 corpora (calibonly, 9bench, ifheavy) × 3 methods (EAC, KD, EAC+KD) = 9 builds
# Plus 3 anchors (A2, A2_RKD, pes1_10) full benches
# Plus gated full bench on 21q winners
set -uo pipefail
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=$BM/logs/post_track5_matrix_$TS
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/chain.log
exec > >(tee "$LOG") 2>&1
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"

CONVERT=$BM/tools/llama.cpp/convert_hf_to_gguf.py
QUANT=/workspace/llama.cpp/build/bin/llama-quantize
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
TEACHER=$BM/google/gemma-4-26B-A4B-it
A2=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it
IMATRIX=$BM/models/variants/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it-GGUF/imatrix.dat
DROP_MAP=$BM/repos/omnimergekit/scripts/v6coder_C6v3lcb_62e_fc15_25_p8_drop_map.json
RES_21Q=$BM/eval_results_corpora_21q
RES_FULL=$BM/eval_results_tracks_2_3
mkdir -p "$RES_21Q" "$RES_FULL"

# Tag → corpus mapping
declare -A EAC_CORPUS=(
    [calibonly]=$BM/scripts/eac_corpus_calib_only.txt
    [9bench]=$BM/scripts/eac_corpus_9bench_balanced.txt
    [ifheavy]=$BM/scripts/eac_corpus_ifeval_heavy.txt
)
declare -A KD_CORPUS=(
    [calibonly]=$BM/scripts/router_calib_corpus.jsonl  # calib-only KD corpus same as base for KD (KD already uses jsonl)
    [9bench]=$BM/scripts/router_calib_corpus_9bench_balanced.jsonl
    [ifheavy]=$BM/scripts/router_calib_corpus_ifeval_heavy.jsonl
)

quantize_q6() {
    local src=$1 q6=$2
    local gguf_dir=$(dirname "$q6")
    local f16="${gguf_dir}/$(basename $q6 .gguf | sed s/-Q6_K/-F16/).gguf"
    mkdir -p "$gguf_dir"
    [ -f "$q6" ] && { echo "  [quant skip] $q6 exists"; return 0; }
    "$PY" "$CONVERT" "$src" --outfile "$f16" --outtype f16 2>&1 | tail -3
    local n=$("$PY" -c "from gguf.gguf_reader import GGUFReader; print(len(GGUFReader(\"$f16\").tensors))")
    [ "$n" -lt 600 ] && { rm "$f16"; "$PY" "$CONVERT" "$src" --outfile "$f16" --outtype f16; }
    "$QUANT" --imatrix "$IMATRIX" "$f16" "$q6" Q6_K 2>&1 | tail -3
    rm -f "$f16"
}

apply_eac() {
    local out=$1 corpus=$2 name=$3
    [ -d "$out" ] && [ -f "$out/model-00006-of-00006.safetensors" ] && { echo "  [eac skip] $out"; return 0; }
    cp -a "$A2" "$out"
    "$PY" $BM/scripts/router_eac_calibrate.py \
        --phase both --base-dir "$TEACHER" --variant-dir "$out" \
        --drop-map "$DROP_MAP" --corpus-file "$corpus" \
        --n-seq 128 --seq-len 2048 --batch-size 4 \
        --calib-k 16 --lr 1e-3 --steps 150 \
        --max-gpu-gib 80 --max-cpu-gib 400 \
        --cache-dir $BM/eac_cache_${name} 2>&1 | tail -8
}

apply_kd() {
    local src=$1 out=$2 corpus=$3 name=$4
    [ -d "$out" ] && [ -f "$out/model-00006-of-00006.safetensors" ] && { echo "  [kd skip] $out"; return 0; }
    "$PY" $BM/scripts/router_kd.py \
        --base-dir "$TEACHER" --variant-dir "$src" --out-dir "$out" \
        --teacher-load bf16 --student-load bf16 \
        --teacher-device "{\"\":0}" --student-device "{\"\":1}" \
        --tau 1.0 --lr 1e-5 --max-steps 100 \
        --batch-size 2 --grad-accum 4 \
        --max-seq-len 512 --max-samples 800 \
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

bench_full() {
    local q6=$1 tok=$2 served=$3
    for tpl in humanevalplus_full ifeval_100 multipl_e_100; do
        [ -f "$RES_FULL/$tpl/$served/summary.json" ] && { echo "  [skip] $tpl"; continue; }
        "$PY" "$OMK" --model "$q6" --tokenizer "$tok" --template $tpl --backend llama \
            --served-name "$served" --results-dir "$RES_FULL" 2>&1 | tail -10
    done
}


# DISK-HYGIENE 2026-05-29: per-cell cleanup so 9-cell matrix doesn't fill 2T root.
cleanup_variant() {
    local src=$1 q6=$2 served=$3
    [ -f "$q6" ] || { echo "  [cleanup skip] no Q6 for $served"; return 1; }
    for tpl in humanevalplus_rum3 ifeval_rum3 multipl_e_rum15; do
        [ -f "$RES_21Q/$tpl/$served/summary.json" ] || { echo "  [cleanup skip] 21q $tpl missing for $served"; return 1; }
    done
    local gguf_dir=$(dirname "$q6")
    cp -n "$src"/tokenizer*.json "$gguf_dir/" 2>/dev/null || true
    cp -n "$src"/chat_template.jinja "$gguf_dir/" 2>/dev/null || true
    cp -n "$src"/config.json "$gguf_dir/" 2>/dev/null || true
    cp -n "$src"/generation_config.json "$gguf_dir/" 2>/dev/null || true
    if [ -d "$src" ]; then
        local sz_before=$(du -shx "$src" 2>/dev/null | cut -f1)
        rm -rf "$src"
        echo "  [cleanup] purged $src ($sz_before)"
    fi
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

echo "[$(date -Iseconds)] === post-Track5 9-matrix chain ==="
echo "Building order: EAC-only first (3), then KD on EAC (3), then KD-only (3) = 9 variants"

# === STAGE 1: build + quant + 21q probe all 9 ===
for label in calibonly 9bench ifheavy; do
    eac_corpus=${EAC_CORPUS[$label]}
    kd_corpus=${KD_CORPUS[$label]}
    
    # EAC-only
    EAC_OUT=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-${label}-it
    EAC_Q6=${EAC_OUT}-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-${label}-it-Q6_K.gguf
    echo "[$(date -Iseconds)] >>> EAC-only $label"
    apply_eac "$EAC_OUT" "$eac_corpus" "eac_$label"
    quantize_q6 "$EAC_OUT" "$EAC_Q6"
    bench_21q "$EAC_Q6" "$EAC_OUT" "a2eac_${label}-62e-fc15_25-p8-s1_0p1_20"
    
    # EAC + KD (reuses EAC-only output)
    RKD_OUT=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-${label}-it
    RKD_Q6=${RKD_OUT}-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-${label}-it-Q6_K.gguf
    echo "[$(date -Iseconds)] >>> EAC+KD $label"
    apply_kd "$EAC_OUT" "$RKD_OUT" "$kd_corpus" "rkd_$label"
    quantize_q6 "$RKD_OUT" "$RKD_Q6"
    bench_21q "$RKD_Q6" "$RKD_OUT" "a2rkd_${label}-62e-fc15_25-p8-s1_0p1_20"

    # cleanup both safetensors variants + EAC cache after both benches complete
    cleanup_variant "$EAC_OUT" "$EAC_Q6" "a2eac_${label}-62e-fc15_25-p8-s1_0p1_20"
    cleanup_variant "$RKD_OUT" "$RKD_Q6" "a2rkd_${label}-62e-fc15_25-p8-s1_0p1_20"
    cleanup_eac_cache "eac_$label"
    df -h / | tail -1
done

for label in calibonly 9bench ifheavy; do
    kd_corpus=${KD_CORPUS[$label]}
    KDO_OUT=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-${label}-it
    KDO_Q6=${KDO_OUT}-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-${label}-it-Q6_K.gguf
    echo "[$(date -Iseconds)] >>> KD-only $label (on raw A2, no EAC)"
    apply_kd "$A2" "$KDO_OUT" "$kd_corpus" "kdonly_$label"
    quantize_q6 "$KDO_OUT" "$KDO_Q6"
    bench_21q "$KDO_Q6" "$KDO_OUT" "a2kdonly_${label}-62e-fc15_25-p8-s1_0p1_20"

    cleanup_variant "$KDO_OUT" "$KDO_Q6" "a2kdonly_${label}-62e-fc15_25-p8-s1_0p1_20"
    df -h / | tail -1
done

# === STAGE 2: anchor full benches ===
echo
echo "[$(date -Iseconds)] === STAGE 2: anchor full benches ==="
A2_Q6=$BM/models/variants/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it-Q6_K.gguf
A2_RKD=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-it
A2_RKD_Q6=${A2_RKD}-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-it-Q6_K.gguf
PES10=$BM/google/gemma-4-A4B-62e-fc15_25-p8-pes1_10-it
PES10_Q6=${PES10}-GGUF/gemma-4-A4B-62e-fc15_25-p8-pes1_10-it-Q6_K.gguf
bench_full "$A2_Q6" "$A2" "a2-62e-fc15_25-p8-s1_0p1_20"
bench_full "$A2_RKD_Q6" "$A2_RKD" "a2rkd-62e-fc15_25-p8-s1_0p1_20"
bench_full "$PES10_Q6" "$PES10" "pes1_10-62e-fc15_25-p8"

# === STAGE 3: 21q-gated full bench on winners ===
echo
echo "[$(date -Iseconds)] === STAGE 3: gated full bench on 21q winners ==="
A2_IF=0.333
for label in calibonly 9bench ifheavy; do
    for method in eac rkd kdonly; do
        served="a2${method}_${label}-62e-fc15_25-p8-s1_0p1_20"
        if_score=$(jq -r ".score // 0" "$RES_21Q/ifeval_rum3/$served/summary.json" 2>/dev/null || echo 0)
        he_score=$(jq -r ".score // 0" "$RES_21Q/humanevalplus_rum3/$served/summary.json" 2>/dev/null || echo 0)
        mp_score=$(jq -r ".score // 0" "$RES_21Q/multipl_e_rum15/$served/summary.json" 2>/dev/null || echo 0)
        # Run full bench if: IFEval preserves A2 OR HE+ ≥ 0.333 OR MPE > 0.20
        run_full=0
        awk "BEGIN{exit ($if_score >= 0.333) ? 0 : 1}" && run_full=1
        awk "BEGIN{exit ($he_score >= 0.333) ? 0 : 1}" && run_full=1
        awk "BEGIN{exit ($mp_score > 0.20) ? 0 : 1}" && run_full=1
        if [ $run_full -eq 1 ]; then
            out_dir=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-${method}-${label}-it
            q6=${out_dir}-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-${method}-${label}-it-Q6_K.gguf
            # 2026-05-29 FIX: tokenizer dir = GGUF dir (cleanup_variant purged bf16 source)
            echo "[$(date -Iseconds)] >>> full bench $served (HE+=$he_score IFEval=$if_score MPE=$mp_score)"
            bench_full "$q6" "${out_dir}-GGUF" "$served"
        else
            echo "  [gate skip] $served (HE+=$he_score IFEval=$if_score MPE=$mp_score)"
        fi
    done
done

# === FINAL SUMMARY TABLE ===
echo
echo "[$(date -Iseconds)] === FINAL SUMMARY ==="
printf "%-50s | 21q-HE+ | 21q-IF | 21q-MPE | F-HE+ | F-IF100 | F-MPE100\n" "model"
echo "-------------------------------------------------------------------------------------------------------"
# Anchors first
for n in a2-62e-fc15_25-p8-s1_0p1_20 a2rkd-62e-fc15_25-p8-s1_0p1_20 a2kdonly-62e-fc15_25-p8-s1_0p1_20 pes1_10-62e-fc15_25-p8 a2eac-62e-fc15_25-p8-s1_0p1_20; do
    h21=$(jq -r ".score // \"-\"" "$BM/eval_results_a2_21q_validation/humanevalplus_rum3/$n/summary.json" 2>/dev/null || \
          jq -r ".score // \"-\"" "$RES_21Q/humanevalplus_rum3/$n/summary.json" 2>/dev/null || echo "-")
    [ "$h21" = "-" ] && h21=$(jq -r ".score // \"-\"" "$BM/eval_results_a2_eac_21q_validation/humanevalplus_rum3/$n/summary.json" 2>/dev/null || \
                              jq -r ".score // \"-\"" "$BM/eval_results_a2_kdonly_21q_validation/humanevalplus_rum3/$n/summary.json" 2>/dev/null || \
                              jq -r ".score // \"-\"" "$BM/eval_results_a2_rkd_21q_validation/humanevalplus_rum3/$n/summary.json" 2>/dev/null || echo "-")
    fh=$(jq -r ".score // \"-\"" "$RES_FULL/humanevalplus_full/$n/summary.json" 2>/dev/null || echo "-")
    fi_=$(jq -r ".score // \"-\"" "$RES_FULL/ifeval_100/$n/summary.json" 2>/dev/null || echo "-")
    fm=$(jq -r ".score // \"-\"" "$RES_FULL/multipl_e_100/$n/summary.json" 2>/dev/null || echo "-")
    printf "%-50s | %-7s | %-6s | %-7s | %-5s | %-7s | %-8s\n" "$n" "$h21" "?" "?" "$fh" "$fi_" "$fm"
done
# New 9 from this run
for label in calibonly 9bench ifheavy; do
    for method in eac rkd kdonly; do
        n="a2${method}_${label}-62e-fc15_25-p8-s1_0p1_20"
        h21=$(jq -r ".score // \"-\"" "$RES_21Q/humanevalplus_rum3/$n/summary.json" 2>/dev/null || echo "-")
        i21=$(jq -r ".score // \"-\"" "$RES_21Q/ifeval_rum3/$n/summary.json" 2>/dev/null || echo "-")
        m21=$(jq -r ".score // \"-\"" "$RES_21Q/multipl_e_rum15/$n/summary.json" 2>/dev/null || echo "-")
        fh=$(jq -r ".score // \"-\"" "$RES_FULL/humanevalplus_full/$n/summary.json" 2>/dev/null || echo "-")
        fi_=$(jq -r ".score // \"-\"" "$RES_FULL/ifeval_100/$n/summary.json" 2>/dev/null || echo "-")
        fm=$(jq -r ".score // \"-\"" "$RES_FULL/multipl_e_100/$n/summary.json" 2>/dev/null || echo "-")
        printf "%-50s | %-7s | %-6s | %-7s | %-5s | %-7s | %-8s\n" "$n" "$h21" "$i21" "$m21" "$fh" "$fi_" "$fm"
    done
done

echo
echo "[$(date -Iseconds)] === DONE ==="
