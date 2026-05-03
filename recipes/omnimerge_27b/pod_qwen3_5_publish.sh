#!/bin/bash
# Phase 3 — publish the merged Omnimerge to HuggingFace.
# Run ONLY after reviewing HumanEval results in pod_qwen3_5_merge_and_eval.sh output.
#
# Creates:
#   ManniX-ITA/Qwen3.5-27B-Omnimerge         (HF model)
#   ManniX-ITA/Qwen3.5-27B-Omnimerge-GGUF    (GGUF pack — standard quants via quantize_gguf.py)
#
# Invoke:
#   HF_TOKEN=hf_xxx bash pod_qwen3_5_publish.sh
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN required}"

WORKDIR=/workspace
MERGED=$WORKDIR/merged_omnimerge
REPO=ManniX-ITA/Qwen3.5-27B-Omnimerge
GGUF_REPO=${REPO}-GGUF

# --- Upload HF model ---
echo "=== upload HF model → $REPO ==="
python3 - <<PY
from huggingface_hub import HfApi, create_repo
api = HfApi()
create_repo("$REPO", exist_ok=True, repo_type="model")
api.upload_folder(
    folder_path="$MERGED",
    repo_id="$REPO",
    repo_type="model",
    ignore_patterns=["*.gguf", "*.bin.index.json", ".cache*", "*.log"],
    commit_message="Omnimerge: flat 4-way DARE-TIES of Jackrong + DavidAU + ConicCat + Esper3.1 (Qwen3.5-27B chassis)",
)
print("HF model uploaded to", "$REPO")
PY

# --- Quantize full GGUF pack via existing quantize_gguf.py ---
echo
echo "=== quantize_gguf.py full pack → $GGUF_REPO ==="
HF_TOKEN="$HF_TOKEN" python3 $WORKDIR/quantize_gguf.py \
    --model "$MERGED" \
    --repo "$GGUF_REPO" \
    --base-model-id "$REPO" \
    --cal-data $WORKDIR/calibration_datav5.txt \
    --sanity-check \
    --exclude CD-Q6_K,CD-Q5_K_M,CD-Q4_K_M,CD-Q3_K_M,CD-Q2_K \
    --hf-token "$HF_TOKEN" 2>&1 | tee $WORKDIR/omnimerge_quantize.log

echo
echo "===== DONE ====="
echo "  HF:   https://huggingface.co/$REPO"
echo "  GGUF: https://huggingface.co/$GGUF_REPO"
