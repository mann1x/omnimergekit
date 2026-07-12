#!/bin/bash
# After A2_RKD safetensors are saved by force_save_a2_rkd.sh,
# convert to F16 GGUF → quantize Q6_K (reuse A2 imatrix) → run 21q probe
# Author: claude opus 4.7  2026-05-29
set -uo pipefail
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=$BM/logs/a2_rkd_chain_$TS
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/chain.log
exec > >(tee "$LOG") 2>&1

A2_RKD=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-it
GGUF_DIR=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-it-GGUF
F16=$GGUF_DIR/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-it-F16.gguf
Q6=$GGUF_DIR/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-it-Q6_K.gguf
IMATRIX=$BM/models/variants/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it-GGUF/imatrix.dat
RES=$BM/eval_results_a2_rkd_21q_validation
NAME="a2rkd-62e-fc15_25-p8-s1_0p1_20"

CONVERT=$BM/tools/llama.cpp/convert_hf_to_gguf.py
QUANT=/workspace/llama.cpp/build/bin/llama-quantize
OMK=/shared/dev/omnimergekit/eval/omk_eval.py
[ -f "$OMK" ] || OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py

echo "[$(date -Iseconds)] === A2_RKD chain ==="
echo "  src:    $A2_RKD"
echo "  f16:    $F16"
echo "  q6_k:   $Q6"
echo "  imat:   $IMATRIX"
echo "  res:    $RES"
echo

# Wait for save
for i in $(seq 1 600); do
    if [ -f "$A2_RKD/model-00006-of-00006.safetensors" ]; then
        echo "  A2_RKD safetensors landed (T+${i}0s wait)"
        break
    fi
    sleep 10
done
[ -f "$A2_RKD/model-00006-of-00006.safetensors" ] || { echo "FATAL: A2_RKD never saved" >&2; exit 2; }
echo "  A2_RKD contents:"
ls "$A2_RKD" | head -10
du -sh "$A2_RKD"

mkdir -p "$GGUF_DIR"

echo
echo "[$(date -Iseconds)] === Step 1: convert HF → F16 GGUF ==="
if [ ! -f "$F16" ]; then
    "$PY" "$CONVERT" "$A2_RKD" --outfile "$F16" --outtype f16 2>&1 | tail -10
    [ -f "$F16" ] || { echo "FATAL: F16 GGUF not created" >&2; exit 3; }
    echo "  F16 created: $(du -sh "$F16" | cut -f1)"
else
    echo "  F16 already exists, skip"
fi

echo
echo "[$(date -Iseconds)] === Step 2: quantize F16 → Q6_K with A2 imatrix ==="
if [ ! -f "$Q6" ]; then
    if [ ! -f "$IMATRIX" ]; then
        echo "  WARN: no imatrix found, quantizing without imatrix"
        "$QUANT" "$F16" "$Q6" Q6_K 2>&1 | tail -10
    else
        "$QUANT" --imatrix "$IMATRIX" "$F16" "$Q6" Q6_K 2>&1 | tail -10
    fi
    [ -f "$Q6" ] || { echo "FATAL: Q6_K not created" >&2; exit 4; }
    echo "  Q6_K created: $(du -sh "$Q6" | cut -f1)"
else
    echo "  Q6_K already exists, skip"
fi

echo
echo "[$(date -Iseconds)] === Step 3: 21q probe ==="
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$RES"
for tpl in humanevalplus_rum3 ifeval_rum3 multipl_e_rum15; do
    echo "  >>> $tpl"
    "$PY" "$OMK" \
        --model "$Q6" \
        --tokenizer "$A2_RKD" \
        --template "$tpl" \
        --backend llama \
        --served-name "$NAME" \
        --results-dir "$RES" 2>&1 | tail -60
    echo "  <<< $tpl rc=$?"
done

echo
echo "[$(date -Iseconds)] === A2_RKD chain complete ==="
for tpl in humanevalplus_rum3 ifeval_rum3 multipl_e_rum15; do
    s=$(jq -r '.score // "null"' "$RES/$tpl/$NAME/summary.json" 2>/dev/null || echo "MISSING")
    printf '%-22s score= %s\n' "$tpl" "$s"
done

# Optional: clean up F16 to save disk (Q6_K is the only one we need)
echo
echo "[$(date -Iseconds)] === cleanup ==="
if [ -f "$Q6" ] && [ -f "$F16" ]; then
    rm -f "$F16"
    echo "  removed F16 (Q6_K preserved)"
fi
