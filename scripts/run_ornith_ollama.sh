#!/usr/bin/env bash
# Ornith -> ollama.com, BOTH arms, TEXT + VISION tags.
#
# Vision is included because it was PROVEN, not assumed: mmproj-Qwen3.6-27B-A3B-Coder-F16
# loaded against Ornith-1.5-27B-A3B-Coder-Q6_K_L in llama-server read a 5-band
# non-guessable test image 5/5 correct, while the no-image control arm correctly said it
# saw no image (/srv/ml/scripts/vision_probe2.py, 2026-09-09).
#
# GGUF_STEM follows the REPO NAME minus -MTP-GGUF. Never an arm codename (184e/P6).
set -uo pipefail
eval "$(grep -m1 '^export HF_TOKEN=' ~/.bashrc)"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be exported}"
S=/srv/ml/scripts/omk_ollama/ollama_publish_tiers.sh
PYBIN=/srv/ml/envs/envs/omnimergekit/bin/python
MM=/mnt/sdc/ream-work/mmproj/mmproj-Qwen3.6-27B-A3B-Coder-F16.gguf
TIERS_COMMON="Q8_0 Q6_K_L Q6_K Q5_K_L Q5_K_M Q5_K_S Q4_K_L Q4_K_S IQ4_NL IQ4_XS Q3_K_XL Q3_K_L Q3_K_M Q3_K_S IQ3_M Q2_K_L IQ2_M IQ2_XS"

echo "===== ARM 1/2: Coder (p0) ====="
PY=$PYBIN MMPROJ=$MM \
HF_REPO=ManniX-ITA/Ornith-1.5-27B-A3B-Coder-MTP-GGUF \
OL_BASE=mannix/ornith-1.5-27b-a3b-coder \
GGUF_STEM=Ornith-1.5-27B-A3B-Coder \
TIERS="$TIERS_COMMON Q4_K_M" LATEST_TIER=Q4_K_M \
WORK=/mnt/sdc/ornith_ol/pub_coder STAGE=/mnt/sdc/ornith_ol/stage \
bash "$S"
echo "===== ARM 1 rc=$? ====="

echo "===== ARM 2/2: CoderX (p6) ====="
PY=$PYBIN MMPROJ=$MM \
HF_REPO=ManniX-ITA/Ornith-1.5-27B-A3B-CoderX-MTP-GGUF \
OL_BASE=mannix/ornith-1.5-27b-a3b-coderx \
GGUF_STEM=Ornith-1.5-27B-A3B-CoderX \
TIERS="$TIERS_COMMON" LATEST_TIER=Q4_K_L \
WORK=/mnt/sdc/ornith_ol/pub_coderx STAGE=/mnt/sdc/ornith_ol/stage \
bash "$S"
echo "===== ARM 2 rc=$? ====="
echo "ORNITH_OLLAMA_PUBLISH_DONE"
