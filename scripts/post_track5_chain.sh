#!/bin/bash
# Post-Track5 chain: EAC corpus ablation (21q-gated) + Track 8 + anchor benches
# Author: claude opus 4.7  2026-05-29
set -uo pipefail
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=$BM/logs/post_track5_$TS
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

quantize_q6() {
    local src=$1 q6=$2 gguf_dir=$3
    local f16="$gguf_dir/$(basename $q6 .gguf | sed s/-Q6_K/-F16/).gguf"
    if [ ! -f "$q6" ]; then
        "$PY" "$CONVERT" "$src" --outfile "$f16" --outtype f16 2>&1 | tail -3
        local n=$("$PY" -c "from gguf.gguf_reader import GGUFReader; print(len(GGUFReader(\"$f16\").tensors))")
        [ "$n" -lt 600 ] && { rm "$f16"; "$PY" "$CONVERT" "$src" --outfile "$f16" --outtype f16; }
        "$QUANT" --imatrix "$IMATRIX" "$f16" "$q6" Q6_K 2>&1 | tail -3
        rm -f "$f16"
    fi
}

bench_21q() {
    local q6=$1 tok=$2 served=$3
    for tpl in humanevalplus_rum3 ifeval_rum3 multipl_e_rum15; do
        if [ -f "$RES_21Q/$tpl/$served/summary.json" ]; then continue; fi
        "$PY" "$OMK" --model "$q6" --tokenizer "$tok" --template $tpl --backend llama \
            --served-name "$served" --results-dir "$RES_21Q" 2>&1 | tail -10
    done
    echo "21q results for $served:"
    for tpl in humanevalplus_rum3 ifeval_rum3 multipl_e_rum15; do
        s=$(jq -r ".score // \"null\"" "$RES_21Q/$tpl/$served/summary.json" 2>/dev/null || echo MISS)
        printf "  %-22s %s\n" "$tpl" "$s"
    done
}

bench_full() {
    local q6=$1 tok=$2 served=$3
    for tpl in humanevalplus_full ifeval_100 multipl_e_100; do
        if [ -f "$RES_FULL/$tpl/$served/summary.json" ]; then
            echo "  [skip] $tpl already done"
            continue
        fi
        "$PY" "$OMK" --model "$q6" --tokenizer "$tok" --template $tpl --backend llama \
            --served-name "$served" --results-dir "$RES_FULL" 2>&1 | tail -10
    done
    echo "FULL results for $served:"
    for tpl in humanevalplus_full ifeval_100 multipl_e_100; do
        s=$(jq -r ".score // \"null\"" "$RES_FULL/$tpl/$served/summary.json" 2>/dev/null || echo MISS)
        printf "  %-22s %s\n" "$tpl" "$s"
    done
}

apply_eac() {
    local out=$1 corpus=$2 name=$3
    if [ -d "$out" ] && [ -f "$out/model-00006-of-00006.safetensors" ]; then
        echo "  [skip] $out already built"
        return 0
    fi
    cp -a "$A2" "$out"
    "$PY" $BM/scripts/router_eac_calibrate.py \
        --phase both --base-dir "$TEACHER" --variant-dir "$out" \
        --drop-map "$DROP_MAP" --corpus-file "$corpus" \
        --n-seq 128 --seq-len 2048 --batch-size 4 \
        --calib-k 16 --lr 1e-3 --steps 150 \
        --max-gpu-gib 80 --max-cpu-gib 400 \
        --cache-dir $BM/eac_cache_${name} 2>&1 | tail -10
}

apply_kd_alone() {
    local out=$1 corpus=$2 name=$3
    if [ -d "$out" ] && [ -f "$out/model-00006-of-00006.safetensors" ]; then
        echo "  [skip] $out already built"; return 0
    fi
    "$PY" $BM/scripts/router_kd.py \
        --base-dir "$TEACHER" --variant-dir "$A2" --out-dir "$out" \
        --teacher-load bf16 --student-load bf16 \
        --teacher-device "{\"\":0}" --student-device "{\"\":1}" \
        --tau 1.0 --lr 1e-5 --max-steps 100 \
        --batch-size 2 --grad-accum 4 \
        --max-seq-len 512 --max-samples 800 \
        --corpus-file "$corpus" \
        --checkpoint-dir $LOG_DIR/ckpt_${name} \
        --canary-file $BM/scripts/ifeval_rumination_canaries.json \
        --no-canary 2>&1 | tail -10
}

echo "[$(date -Iseconds)] === post-Track5 chain start ==="

# === STAGE 1: EAC corpus validation suite (3 variants, 21q-gated) ===
echo "[$(date -Iseconds)] === STAGE 1: EAC corpus ablation ==="

for variant in calibonly:eac_corpus_calib_only.txt:a2eac_calibonly \
               9bench:eac_corpus_9bench_balanced.txt:a2eac_9bench \
               ifheavy:eac_corpus_ifeval_heavy.txt:a2eac_ifheavy; do
    label=$(echo $variant | cut -d: -f1)
    corpus=$BM/scripts/$(echo $variant | cut -d: -f2)
    served=$(echo $variant | cut -d: -f3)-62e-fc15_25-p8-s1_0p1_20
    out=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-${label}-it
    gguf_dir=${out}-GGUF
    q6=$gguf_dir/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-${label}-it-Q6_K.gguf
    echo "[$(date -Iseconds)] >>> EAC variant: $label"
    apply_eac "$out" "$corpus" "eac_$label"
    mkdir -p "$gguf_dir"
    quantize_q6 "$out" "$q6" "$gguf_dir"
    bench_21q "$q6" "$out" "$served"
done

# Decision: if any EAC variant preserves IFEval, log it. Run full bench on best.
echo
echo "[$(date -Iseconds)] === STAGE 1 GATE: pick best EAC variant for full bench ==="
BEST_VAR=""
BEST_IF=0
for variant in calibonly:a2eac_calibonly 9bench:a2eac_9bench ifheavy:a2eac_ifheavy; do
    label=$(echo $variant | cut -d: -f1)
    served=$(echo $variant | cut -d: -f2)-62e-fc15_25-p8-s1_0p1_20
    if_score=$(jq -r ".score // 0" "$RES_21Q/ifeval_rum3/$served/summary.json" 2>/dev/null || echo 0)
    he_score=$(jq -r ".score // 0" "$RES_21Q/humanevalplus_rum3/$served/summary.json" 2>/dev/null || echo 0)
    mp_score=$(jq -r ".score // 0" "$RES_21Q/multipl_e_rum15/$served/summary.json" 2>/dev/null || echo 0)
    printf "  EAC-%s: HE+=%s IFEval=%s MPE=%s\n" "$label" "$he_score" "$if_score" "$mp_score"
    # Trigger full bench if IFEval ≥ 0.333
    if awk "BEGIN{exit ($if_score >= 0.333) ? 0 : 1}"; then
        echo "    -> EAC-$label PRESERVES IFEval, queuing full bench"
        out=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-${label}-it
        q6_dir=${out}-GGUF
        q6=$q6_dir/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-${label}-it-Q6_K.gguf
        bench_full "$q6" "$out" "$served"
    fi
done

# === STAGE 2: Track 8 — KD-only on A2 with IFEval-heavy corpus ===
echo
echo "[$(date -Iseconds)] === STAGE 2: Track 8 KD-only IFEval-heavy ==="
T8_OUT=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-ifheavy-it
apply_kd_alone "$T8_OUT" "$BM/scripts/router_calib_corpus_ifeval_heavy.jsonl" "t8_kdonly_ifheavy"
T8_GGUF=${T8_OUT}-GGUF
mkdir -p "$T8_GGUF"
T8_Q6=$T8_GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-ifheavy-it-Q6_K.gguf
quantize_q6 "$T8_OUT" "$T8_Q6" "$T8_GGUF"
bench_21q "$T8_Q6" "$T8_OUT" "a2kdonly-ifheavy-62e-fc15_25-p8-s1_0p1_20"
bench_full "$T8_Q6" "$T8_OUT" "a2kdonly-ifheavy-62e-fc15_25-p8-s1_0p1_20"

# === STAGE 3: Anchor full benches (A2, A2_RKD, pes1_10) ===
echo
echo "[$(date -Iseconds)] === STAGE 3: Anchor full benches ==="
# A2
A2_Q6=$BM/models/variants/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it-Q6_K.gguf
bench_full "$A2_Q6" "$A2" "a2-62e-fc15_25-p8-s1_0p1_20"
# A2_RKD
A2_RKD=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-it
A2_RKD_Q6=${A2_RKD}-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-it-Q6_K.gguf
bench_full "$A2_RKD_Q6" "$A2_RKD" "a2rkd-62e-fc15_25-p8-s1_0p1_20"
# pes1_10 
PES10=$BM/google/gemma-4-A4B-62e-fc15_25-p8-pes1_10-it
PES10_Q6=${PES10}-GGUF/gemma-4-A4B-62e-fc15_25-p8-pes1_10-it-Q6_K.gguf
bench_full "$PES10_Q6" "$PES10" "pes1_10-62e-fc15_25-p8"

echo
echo "[$(date -Iseconds)] === ALL STAGES DONE — SUMMARY TABLE ==="
printf "%-40s | %-10s | %-10s | %-10s\n" "model" "HE+164" "IFEval100" "MPE100"
echo "------------------------------------------------------------------------------------"
for n in a2-62e-fc15_25-p8-s1_0p1_20 a2rkd-62e-fc15_25-p8-s1_0p1_20 \
         a2kdonly-62e-fc15_25-p8-s1_0p1_20 a2kdonly-ifheavy-62e-fc15_25-p8-s1_0p1_20 \
         pes1_10-62e-fc15_25-p8 \
         a2eac_calibonly-62e-fc15_25-p8-s1_0p1_20 a2eac_9bench-62e-fc15_25-p8-s1_0p1_20 \
         a2eac_ifheavy-62e-fc15_25-p8-s1_0p1_20; do
    he=$(jq -r ".score // \"null\"" "$RES_FULL/humanevalplus_full/$n/summary.json" 2>/dev/null || echo MISS)
    if=$(jq -r ".score // \"null\"" "$RES_FULL/ifeval_100/$n/summary.json" 2>/dev/null || echo MISS)
    mp=$(jq -r ".score // \"null\"" "$RES_FULL/multipl_e_100/$n/summary.json" 2>/dev/null || echo MISS)
    printf "%-40s | %-10s | %-10s | %-10s\n" "$n" "$he" "$if" "$mp"
done
