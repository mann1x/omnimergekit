#!/bin/bash
# Quantize 109e, 98e-hybrid, 120e-hybrid v3 to Q6_K and run the 11q
# definitive test on each, with the locked methodology.
#
# Per model: convert HF -> F16 GGUF (~2 min, ~45 GB), quantize -> Q6_K
# (~2 min, ~18 GB), delete F16 intermediate, run 11q eval (~30 min).
#
# Order: 109e (reference) -> 98e-hybrid (winner candidate) -> 120e v3 (experimental)
set -euo pipefail

REPO="/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models"
cd "$REPO"

eval "$(/root/anaconda3/bin/conda shell.bash hook)"
conda activate lightseek

LLAMA_CONVERT="/opt/llama.cpp/convert_hf_to_gguf.py"
LLAMA_QUANTIZE="/opt/llama.cpp/build/bin/llama-quantize"

# (label, hf_dir, output_quant_path)
MODELS=(
    "109e:google/gemma-4-A4B-109e:google/gemma-4-A4B-109e-Q6_K.gguf"
    "98e_hybrid:google/gemma-4-A4B-98e-hybrid:google/gemma-4-A4B-98e-hybrid-Q6_K.gguf"
    "120e_v3:google/gemma-4-A4B-120e-hybrid:google/gemma-4-A4B-120e-hybrid-Q6_K.gguf"
)

PIPELINE_START=$(date +%s)
echo "===== pruned Q6_K pipeline start: $(date) ====="

for entry in "${MODELS[@]}"; do
    LABEL="${entry%%:*}"
    REST="${entry#*:}"
    HF_DIR="${REST%%:*}"
    Q6K="${REST##*:}"
    F16="${Q6K%-Q6_K.gguf}-F16.gguf"

    echo
    echo "===================================================================="
    echo "===== Processing $LABEL ====="
    echo "  HF dir: $HF_DIR"
    echo "  F16:    $F16"
    echo "  Q6_K:   $Q6K"
    echo "  start:  $(date)"
    echo "===================================================================="

    if [[ -f "$Q6K" ]]; then
        echo "  Q6_K already exists, skipping quantization"
    else
        echo "  -> converting HF to F16..."
        python3 "$LLAMA_CONVERT" "$HF_DIR" --outfile "$F16" --outtype f16 2>&1 | tail -3
        ls -lh "$F16"

        echo "  -> quantizing F16 to Q6_K..."
        "$LLAMA_QUANTIZE" "$F16" "$Q6K" Q6_K 2>&1 | tail -5
        ls -lh "$Q6K"

        echo "  -> deleting F16 intermediate..."
        rm "$F16"
    fi

    echo
    echo "  -> running 11q eval on $LABEL..."
    ./scripts/eval_6q.sh "${LABEL}_Q6K_def" "$Q6K"
    echo "  $LABEL eval done at $(date)"
done

echo
echo "===================================================================="
PIPELINE_END=$(date +%s)
PIPELINE_MINS=$(( (PIPELINE_END - PIPELINE_START) / 60 ))
echo "===== pipeline complete: $(date) — total ${PIPELINE_MINS} min ====="
