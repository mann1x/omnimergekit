#!/bin/bash
# Tracks 2 + 3 master chain — EAC+KD ablation study on bs2
# Author: claude opus 4.7  2026-05-29
# Builds 4 new models, full HE+164/IFEval100/MPE100 bench on 5 cohort members.
# Sequential single-GPU plan: ~5 hr wall on 2× PRO 6000.
set -uo pipefail
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=$BM/logs/tracks_2_3_$TS
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/chain.log
exec > >(tee "$LOG") 2>&1
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"

# === Tools ===
CONVERT=$BM/tools/llama.cpp/convert_hf_to_gguf.py
QUANT=/workspace/llama.cpp/build/bin/llama-quantize
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
PES=$BM/scripts/router_per_expert_rescale.py
EAC_CORPUS=$BM/scripts/eac_corpus_wiki2_plus_calib.txt
KD_CORPUS=$BM/scripts/router_calib_corpus.jsonl
DROP_MAP=$BM/repos/omnimergekit/scripts/v6coder_C6v3lcb_62e_fc15_25_p8_drop_map.json
IMATRIX=$BM/models/variants/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it-GGUF/imatrix.dat
TEACHER=$BM/google/gemma-4-26B-A4B-it
A2=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it

# === Track 2/3 model dirs ===
RAW62=$BM/google/gemma-4-A4B-62e-fc15_25-p8-raw-it
RAW62_EACKD=$BM/google/gemma-4-A4B-62e-fc15_25-p8-raw-eackd-it
PES10=$BM/google/gemma-4-A4B-62e-fc15_25-p8-pes1_10-it
PES10_EACKD=$BM/google/gemma-4-A4B-62e-fc15_25-p8-pes1_10-eackd-it
A2_RKD=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-it
A2_EAC=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-it

# === Results dir ===
RES=$BM/eval_results_tracks_2_3
mkdir -p "$RES"

run_bench() {
    local name=$1 model_dir=$2 q6=$3
    for tpl in humanevalplus_full ifeval_100 multipl_e_100; do
        local sub="$RES/$tpl/$name"
        if [ -f "$sub/summary.json" ]; then
            local s=$(jq -r ".score // \"null\"" "$sub/summary.json" 2>/dev/null || echo "MISSING")
            echo "  [skip] $name/$tpl already done — score=$s"
            continue
        fi
        echo "  [$(date -Iseconds)] >>> $name / $tpl"
        "$PY" "$OMK" --model "$q6" --tokenizer "$model_dir" \
            --template "$tpl" --backend llama \
            --served-name "$name" --results-dir "$RES" 2>&1 | tail -25
        echo "  [$(date -Iseconds)] <<< $name / $tpl"
    done
}

quantize_q6() {
    local src=$1 q6=$2
    local gguf_dir="$(dirname "$q6")"
    local f16="${gguf_dir}/$(basename "$q6" .gguf | sed "s/-Q6_K/-F16/").gguf"
    mkdir -p "$gguf_dir"
    if [ ! -f "$q6" ]; then
        if [ ! -f "$f16" ]; then
            echo "  [quant] HF→F16"
            "$PY" "$CONVERT" "$src" --outfile "$f16" --outtype f16 2>&1 | tail -3
        fi
        local n=$("$PY" -c "from gguf.gguf_reader import GGUFReader; print(len(GGUFReader(\"$f16\").tensors))")
        if [ "$n" -lt 600 ]; then
            echo "  [quant] F16 truncated ($n) — retry"
            rm -f "$f16"
            "$PY" "$CONVERT" "$src" --outfile "$f16" --outtype f16 2>&1 | tail -3
        fi
        echo "  [quant] F16→Q6_K"
        "$QUANT" --imatrix "$IMATRIX" "$f16" "$q6" Q6_K 2>&1 | tail -3
        rm -f "$f16"
    fi
    echo "  [quant] Q6_K: $(ls -la "$q6" | awk "{print \$5}") bytes"
}

apply_eac_kd() {
    local src=$1 out=$2 name=$3
    echo "  [$(date -Iseconds)] EAC+KD on $name"
    if [ -d "$out" ] && [ -f "$out/model-00006-of-00006.safetensors" ]; then
        echo "  [skip] $out already present"
        return 0
    fi
    # Phase A: EAC in-place on copy
    local eac_dir="${out}.eac_tmp"
    [ ! -d "$eac_dir" ] && cp -a "$src" "$eac_dir"
    "$PY" $BM/scripts/router_eac_calibrate.py \
        --phase both --base-dir "$TEACHER" --variant-dir "$eac_dir" \
        --drop-map "$DROP_MAP" --corpus-file "$EAC_CORPUS" \
        --n-seq 128 --seq-len 2048 --batch-size 4 \
        --calib-k 16 --lr 1e-3 --steps 150 \
        --max-gpu-gib 80 --max-cpu-gib 400 \
        --cache-dir $BM/eac_cache_${name} 2>&1 | tail -10
    # Phase B: Router-KD with --no-canary (force save; we evaluate post-hoc)
    "$PY" $BM/scripts/router_kd.py \
        --base-dir "$TEACHER" --variant-dir "$eac_dir" --out-dir "$out" \
        --teacher-load bf16 --student-load bf16 \
        --teacher-device '{"":0}' --student-device '{"":1}' \
        --tau 1.0 --lr 1e-5 --max-steps 100 \
        --batch-size 2 --grad-accum 4 \
        --max-seq-len 512 --max-samples 800 \
        --corpus-file "$KD_CORPUS" \
        --checkpoint-dir $LOG_DIR/ckpt_${name} \
        --canary-file $BM/scripts/ifeval_rumination_canaries.json \
        --no-canary 2>&1 | tail -10
    rm -rf "$eac_dir"
}

echo "[$(date -Iseconds)] === Tracks 2+3 master chain start ==="

# === Track 2a: build truly-raw 62e (shared α=1.0, no PES) ===
echo
echo "[$(date -Iseconds)] === Track 2a: build 62e_raw ==="
if [ ! -d "$RAW62" ]; then
    cp -a "$A2" "$RAW62"
    # Restore PES-rescale backups: A2 shards before PES α=1.20 applied
    # = "shared α=1.0 (inverted) + no PES" = truly raw 62e
    for i in 1 2 3 4 5 6; do
        shard="model-0000${i}-of-00006.safetensors"
        src="${A2}/${shard}.pre_per_expert_rescale"
        dst="${RAW62}/${shard}"
        if [ -f "$src" ]; then
            cp "$src" "${dst}.restore_tmp"
            mv "${dst}.restore_tmp" "$dst"
            echo "  restored shard $i"
        else
            echo "  MISSING $src — cannot restore"
            continue
        fi
    done
    # Remove backup files inside RAW62 to save disk
    rm -f "$RAW62"/*.pre_*
else
    echo "  $RAW62 already exists, skip"
fi

# Quantize raw62 Q6_K
RAW62_GGUF=$BM/google/gemma-4-A4B-62e-fc15_25-p8-raw-it-GGUF
mkdir -p "$RAW62_GGUF"
quantize_q6 "$RAW62" "$RAW62_GGUF/gemma-4-A4B-62e-fc15_25-p8-raw-it-Q6_K.gguf"

# === Track 3a: build pes1_10 (shared α=1.0 + PES α=1.10) ===
echo
echo "[$(date -Iseconds)] === Track 3a: build pes1_10 ==="
if [ ! -d "$PES10" ]; then
    cp -a "$RAW62" "$PES10"
    # Apply PES α=1.10
    "$PY" "$PES" --model-dir "$PES10" --alpha 1.10 2>&1 | tail -5
else
    echo "  $PES10 already exists, skip"
fi
PES10_GGUF=$BM/google/gemma-4-A4B-62e-fc15_25-p8-pes1_10-it-GGUF
mkdir -p "$PES10_GGUF"
quantize_q6 "$PES10" "$PES10_GGUF/gemma-4-A4B-62e-fc15_25-p8-pes1_10-it-Q6_K.gguf"

# === Track 2b: apply EAC+KD to 62e_raw ===
echo
echo "[$(date -Iseconds)] === Track 2b: EAC+KD on 62e_raw ==="
apply_eac_kd "$RAW62" "$RAW62_EACKD" "raw62_eackd"
RAW62_EACKD_GGUF=$BM/google/gemma-4-A4B-62e-fc15_25-p8-raw-eackd-it-GGUF
mkdir -p "$RAW62_EACKD_GGUF"
quantize_q6 "$RAW62_EACKD" "$RAW62_EACKD_GGUF/gemma-4-A4B-62e-fc15_25-p8-raw-eackd-it-Q6_K.gguf"

# === Track 3b: apply EAC+KD to pes1_10 ===
echo
echo "[$(date -Iseconds)] === Track 3b: EAC+KD on pes1_10 ==="
apply_eac_kd "$PES10" "$PES10_EACKD" "pes10_eackd"
PES10_EACKD_GGUF=$BM/google/gemma-4-A4B-62e-fc15_25-p8-pes1_10-eackd-it-GGUF
mkdir -p "$PES10_EACKD_GGUF"
quantize_q6 "$PES10_EACKD" "$PES10_EACKD_GGUF/gemma-4-A4B-62e-fc15_25-p8-pes1_10-eackd-it-Q6_K.gguf"

# === Bench cohort ===
echo
echo "[$(date -Iseconds)] === Full bench cohort (HE+164 + IFEval100 + MPE100) ==="

# Anchor: A2 (= T140)
echo "[$(date -Iseconds)] >>> A2 baseline (anchor)"
A2_Q6=$BM/models/variants/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it-Q6_K.gguf
[ -f "$A2_Q6" ] && run_bench "a2-62e-fc15_25-p8-s1_0p1_20" "$A2" "$A2_Q6" || echo "[skip] A2 Q6_K missing"

# Anchor: A2_RKD (existing)
echo "[$(date -Iseconds)] >>> A2_RKD (anchor)"
A2_RKD_Q6=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-it-Q6_K.gguf
[ -f "$A2_RKD_Q6" ] && run_bench "a2rkd-62e-fc15_25-p8-s1_0p1_20" "$A2_RKD" "$A2_RKD_Q6" || echo "[skip] A2_RKD Q6_K missing"

# Anchor: A2_EAC (Track 1 product)
echo "[$(date -Iseconds)] >>> A2_EAC (anchor)"
A2_EAC_Q6=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-it-Q6_K.gguf
[ -f "$A2_EAC_Q6" ] && run_bench "a2eac-62e-fc15_25-p8-s1_0p1_20" "$A2_EAC" "$A2_EAC_Q6" || echo "[skip] A2_EAC Q6_K missing"

# Track 2a + 2b
run_bench "raw62-62e-fc15_25-p8" "$RAW62" "$RAW62_GGUF/gemma-4-A4B-62e-fc15_25-p8-raw-it-Q6_K.gguf"
run_bench "raw62-eackd-62e-fc15_25-p8" "$RAW62_EACKD" "$RAW62_EACKD_GGUF/gemma-4-A4B-62e-fc15_25-p8-raw-eackd-it-Q6_K.gguf"

# Track 3a + 3b
run_bench "pes1_10-62e-fc15_25-p8" "$PES10" "$PES10_GGUF/gemma-4-A4B-62e-fc15_25-p8-pes1_10-it-Q6_K.gguf"
run_bench "pes1_10-eackd-62e-fc15_25-p8" "$PES10_EACKD" "$PES10_EACKD_GGUF/gemma-4-A4B-62e-fc15_25-p8-pes1_10-eackd-it-Q6_K.gguf"

echo
echo "[$(date -Iseconds)] === SUMMARY TABLE ==="
printf "%-32s | %-12s | %-12s | %-12s\n" "model" "HE+164" "IFEval100" "MPE100"
echo "----------------------------------------------------------------------------------"
for n in a2-62e-fc15_25-p8-s1_0p1_20 a2rkd-62e-fc15_25-p8-s1_0p1_20 a2eac-62e-fc15_25-p8-s1_0p1_20 \
         raw62-62e-fc15_25-p8 raw62-eackd-62e-fc15_25-p8 pes1_10-62e-fc15_25-p8 pes1_10-eackd-62e-fc15_25-p8; do
    he=$(jq -r ".score // \"null\"" "$RES/humanevalplus_full/$n/summary.json" 2>/dev/null || echo MISS)
    if=$(jq -r ".score // \"null\"" "$RES/ifeval_100/$n/summary.json" 2>/dev/null || echo MISS)
    mp=$(jq -r ".score // \"null\"" "$RES/multipl_e_100/$n/summary.json" 2>/dev/null || echo MISS)
    printf "%-32s | %-12s | %-12s | %-12s\n" "$n" "$he" "$if" "$mp"
done

echo "[$(date -Iseconds)] === Tracks 2+3 master chain done ==="
