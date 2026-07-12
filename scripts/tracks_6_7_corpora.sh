#!/bin/bash
# Tracks 6 + 7: EAC+KD on A2 with alternative corpora
# Track 6 = calib-only (no wiki2)
# Track 7 = 9-bench balanced (no wiki2, balanced per-bench)
# Author: claude opus 4.7  2026-05-29
set -uo pipefail
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=$BM/logs/tracks_6_7_$TS
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
RES_FULL=$BM/eval_results_tracks_2_3
RES_21Q=$BM/eval_results_corpora_21q
mkdir -p "$RES_FULL" "$RES_21Q"

run_eac_kd_bench() {
    local trk_name=$1 eac_corpus=$2 kd_corpus=$3 out_dir=$4
    local served=$5
    echo
    echo "[$(date -Iseconds)] ===== Track $trk_name ====="
    echo "  EAC corpus: $eac_corpus"
    echo "  KD  corpus: $kd_corpus"
    echo "  out:        $out_dir"
    if [ -d "$out_dir" ] && [ -f "$out_dir/model-00006-of-00006.safetensors" ]; then
        echo "  [skip] already built"
    else
        local eac_dir="${out_dir}.eac_tmp"
        [ ! -d "$eac_dir" ] && cp -a "$A2" "$eac_dir"
        # EAC
        "$PY" $BM/scripts/router_eac_calibrate.py \
            --phase both --base-dir "$TEACHER" --variant-dir "$eac_dir" \
            --drop-map "$DROP_MAP" --corpus-file "$eac_corpus" \
            --n-seq 128 --seq-len 2048 --batch-size 4 \
            --calib-k 16 --lr 1e-3 --steps 150 \
            --max-gpu-gib 80 --max-cpu-gib 400 \
            --cache-dir $BM/eac_cache_${trk_name} 2>&1 | tail -10
        # KD with --no-canary
        "$PY" $BM/scripts/router_kd.py \
            --base-dir "$TEACHER" --variant-dir "$eac_dir" --out-dir "$out_dir" \
            --teacher-load bf16 --student-load bf16 \
            --teacher-device "{\"\":0}" --student-device "{\"\":1}" \
            --tau 1.0 --lr 1e-5 --max-steps 100 \
            --batch-size 2 --grad-accum 4 \
            --max-seq-len 512 --max-samples 800 \
            --corpus-file "$kd_corpus" \
            --checkpoint-dir $LOG_DIR/ckpt_${trk_name} \
            --canary-file $BM/scripts/ifeval_rumination_canaries.json \
            --no-canary 2>&1 | tail -10
        rm -rf "$eac_dir"
    fi
    # Quantize
    local gguf_dir="${out_dir}-GGUF"
    mkdir -p "$gguf_dir"
    local q6="$gguf_dir/$(basename $out_dir)-Q6_K.gguf"
    local f16="$gguf_dir/$(basename $out_dir)-F16.gguf"
    if [ ! -f "$q6" ]; then
        "$PY" "$CONVERT" "$out_dir" --outfile "$f16" --outtype f16 2>&1 | tail -3
        n=$("$PY" -c "from gguf.gguf_reader import GGUFReader; print(len(GGUFReader(\"$f16\").tensors))")
        [ "$n" -lt 600 ] && { rm "$f16"; "$PY" "$CONVERT" "$out_dir" --outfile "$f16" --outtype f16; }
        "$QUANT" --imatrix "$IMATRIX" "$f16" "$q6" Q6_K 2>&1 | tail -3
        rm -f "$f16"
    fi
    ls -la "$q6"
    # 21q probe
    for tpl in humanevalplus_rum3 ifeval_rum3 multipl_e_rum15; do
        if [ -f "$RES_21Q/$tpl/$served/summary.json" ]; then
            echo "  [skip] 21q/$tpl already done"
            continue
        fi
        echo "  >>> 21q/$tpl"
        "$PY" "$OMK" --model "$q6" --tokenizer "$out_dir" \
            --template "$tpl" --backend llama \
            --served-name "$served" --results-dir "$RES_21Q" 2>&1 | tail -20
    done
    # Full bench
    for tpl in humanevalplus_full ifeval_100 multipl_e_100; do
        if [ -f "$RES_FULL/$tpl/$served/summary.json" ]; then
            echo "  [skip] full/$tpl already done"
            continue
        fi
        echo "  >>> full/$tpl"
        "$PY" "$OMK" --model "$q6" --tokenizer "$out_dir" \
            --template "$tpl" --backend llama \
            --served-name "$served" --results-dir "$RES_FULL" 2>&1 | tail -20
    done
    echo "[$(date -Iseconds)] ===== Track $trk_name done ====="
    for tpl in humanevalplus_rum3 ifeval_rum3 multipl_e_rum15 humanevalplus_full ifeval_100 multipl_e_100; do
        local d=$RES_21Q; [[ $tpl == *_full || $tpl == *_100 ]] && d=$RES_FULL
        s=$(jq -r ".score // \"null\"" "$d/$tpl/$served/summary.json" 2>/dev/null || echo MISS)
        printf "  %-22s score=%s\n" "$tpl" "$s"
    done
}

# Track 6: calib-only EAC (KD corpus unchanged from current)
run_eac_kd_bench \
    "6_calibonly" \
    $BM/scripts/eac_corpus_calib_only.txt \
    $BM/scripts/router_calib_corpus.jsonl \
    $BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-calibonly-it \
    "a2rkd-calibonly-62e-fc15_25-p8-s1_0p1_20"

# Track 7: 9-bench balanced both EAC + KD corpora
run_eac_kd_bench \
    "7_9bench" \
    $BM/scripts/eac_corpus_9bench_balanced.txt \
    $BM/scripts/router_calib_corpus_9bench_balanced.jsonl \
    $BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-9bench-it \
    "a2rkd-9bench-62e-fc15_25-p8-s1_0p1_20"

echo
echo "[$(date -Iseconds)] ===== Tracks 6+7 ALL DONE ====="
