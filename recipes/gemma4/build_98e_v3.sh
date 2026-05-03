#!/bin/bash
# Build + eval 98e v3 from the CLEAN teacher-force map.
# Pipeline: expert_drop → HF to F16 GGUF → Q6_K quantize → cleanup → eval_gpqa_v3.sh
set -euo pipefail

cd /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models

DROP_MAP="scripts/teacher_force_98e_p16_clean.json"
SUFFIX="-v3"
HF_DIR="google/gemma-4-A4B-98e-v3"
F16_GGUF="google/gemma-4-A4B-98e-v3-F16.gguf"
Q6K_GGUF="google/gemma-4-A4B-98e-v3-Q6_K.gguf"

eval "$(/root/anaconda3/bin/conda shell.bash hook)"
conda activate lightseek

echo "===== $(date) 98e v3 BUILD start ====="
echo "  drop map: $DROP_MAP"
echo "  output:   $HF_DIR / $F16_GGUF / $Q6K_GGUF"
echo

# 1) Expert drop HF model
if [[ ! -d "$HF_DIR" ]]; then
    echo "===== [1/4] expert_drop.py ====="
    python3 scripts/expert_drop.py \
        --source-dir google/gemma-4-26B-A4B-it \
        --drop-map "$DROP_MAP" \
        --suffix="$SUFFIX"
else
    echo "===== [1/4] $HF_DIR already exists, skip ====="
fi

# 2) Convert HF to F16 GGUF
if [[ ! -f "$F16_GGUF" ]]; then
    echo
    echo "===== [2/4] convert_hf_to_gguf.py (F16) ====="
    python3 /opt/llama.cpp/convert_hf_to_gguf.py "$HF_DIR" \
        --outfile "$F16_GGUF" --outtype f16
else
    echo "===== [2/4] $F16_GGUF already exists, skip ====="
fi

# 3) Quantize F16 → Q6_K
if [[ ! -f "$Q6K_GGUF" ]]; then
    echo
    echo "===== [3/4] llama-quantize Q6_K ====="
    /opt/llama.cpp/build/bin/llama-quantize "$F16_GGUF" "$Q6K_GGUF" Q6_K
else
    echo "===== [3/4] $Q6K_GGUF already exists, skip ====="
fi

# 4) Cleanup intermediates (keep Q6_K)
echo
echo "===== cleanup intermediates ====="
if [[ -f "$F16_GGUF" ]]; then
    rm "$F16_GGUF" && echo "  removed $F16_GGUF"
fi
if [[ -d "$HF_DIR" ]]; then
    rm -rf "$HF_DIR" && echo "  removed $HF_DIR"
fi

ls -la "$Q6K_GGUF"
echo

# 5) Eval via canonical script (has retry loop from bug-010 fix)
echo
echo "===== [4/4] eval_gpqa_v3.sh ====="
echo "  with MAX_RETRIES=10 for overnight unattended run"
MAX_RETRIES=10 ./scripts/eval_gpqa_v3.sh 98e_v3_Q6K "$Q6K_GGUF"

echo
echo "===== $(date) 98e v3 DONE ====="
