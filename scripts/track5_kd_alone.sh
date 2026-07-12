#!/bin/bash
# Track 5: Router-KD applied directly to A2 (no EAC step first)
# Author: claude opus 4.7  2026-05-29
set -uo pipefail
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=$BM/logs/track5_kd_alone_$TS
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/chain.log
exec > >(tee "$LOG") 2>&1
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"

TEACHER=$BM/google/gemma-4-26B-A4B-it
A2=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it
A2_KDONLY=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-it
KD_CORPUS=$BM/scripts/router_calib_corpus.jsonl
CONVERT=$BM/tools/llama.cpp/convert_hf_to_gguf.py
QUANT=/workspace/llama.cpp/build/bin/llama-quantize
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
IMATRIX=$BM/models/variants/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it-GGUF/imatrix.dat

echo "[$(date -Iseconds)] === Track 5: KD-alone on A2 (no EAC) ==="

# 1. KD directly on A2 (skip EAC step)
if [ ! -d "$A2_KDONLY" ]; then
    "$PY" $BM/scripts/router_kd.py \
        --base-dir "$TEACHER" --variant-dir "$A2" --out-dir "$A2_KDONLY" \
        --teacher-load bf16 --student-load bf16 \
        --teacher-device "{\"\":0}" --student-device "{\"\":1}" \
        --tau 1.0 --lr 1e-5 --max-steps 100 \
        --batch-size 2 --grad-accum 4 \
        --max-seq-len 512 --max-samples 800 \
        --corpus-file "$KD_CORPUS" \
        --checkpoint-dir $LOG_DIR/ckpt \
        --canary-file $BM/scripts/ifeval_rumination_canaries.json \
        --no-canary 2>&1
else
    echo "  $A2_KDONLY exists, skip"
fi

# 2. Quantize Q6_K
GGUF_DIR=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-it-GGUF
F16=$GGUF_DIR/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-it-F16.gguf
Q6=$GGUF_DIR/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-kdonly-it-Q6_K.gguf
NAME=a2kdonly-62e-fc15_25-p8-s1_0p1_20
mkdir -p "$GGUF_DIR"
if [ ! -f "$Q6" ]; then
    "$PY" "$CONVERT" "$A2_KDONLY" --outfile "$F16" --outtype f16 2>&1 | tail -3
    n=$("$PY" -c "from gguf.gguf_reader import GGUFReader; print(len(GGUFReader(\"$F16\").tensors))")
    [ "$n" -lt 600 ] && { rm "$F16"; "$PY" "$CONVERT" "$A2_KDONLY" --outfile "$F16" --outtype f16; }
    "$QUANT" --imatrix "$IMATRIX" "$F16" "$Q6" Q6_K 2>&1 | tail -3
    rm -f "$F16"
fi
ls -la "$Q6"

# 3. 21q probe
RES_21Q=$BM/eval_results_a2_kdonly_21q_validation
for tpl in humanevalplus_rum3 ifeval_rum3 multipl_e_rum15; do
    echo "  >>> $tpl"
    "$PY" "$OMK" --model "$Q6" --tokenizer "$A2_KDONLY" \
        --template "$tpl" --backend llama \
        --served-name "$NAME" --results-dir "$RES_21Q" 2>&1 | tail -20
done

# 4. Full bench (HE+164 + IFEval100 + MPE100) into Tracks 2/3 results dir
RES_FULL=$BM/eval_results_tracks_2_3
for tpl in humanevalplus_full ifeval_100 multipl_e_100; do
    echo "  >>> full/$tpl"
    "$PY" "$OMK" --model "$Q6" --tokenizer "$A2_KDONLY" \
        --template "$tpl" --backend llama \
        --served-name "$NAME" --results-dir "$RES_FULL" 2>&1 | tail -20
done

echo
echo "[$(date -Iseconds)] === Track 5 21q SUMMARY ==="
for tpl in humanevalplus_rum3 ifeval_rum3 multipl_e_rum15; do
    s=$(jq -r ".score // \"null\"" "$RES_21Q/$tpl/$NAME/summary.json" 2>/dev/null || echo MISSING)
    printf "  %-22s score=%s\n" "$tpl" "$s"
done

echo
echo "[$(date -Iseconds)] === Track 5 FULL SUMMARY ==="
for tpl in humanevalplus_full ifeval_100 multipl_e_100; do
    s=$(jq -r ".score // \"null\"" "$RES_FULL/$tpl/$NAME/summary.json" 2>/dev/null || echo MISSING)
    printf "  %-22s score=%s\n" "$tpl" "$s"
done

echo "[$(date -Iseconds)] === Track 5 DONE ==="
