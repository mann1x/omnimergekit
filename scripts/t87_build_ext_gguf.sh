#!/usr/bin/env bash
set -euo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
CONVERT=/srv/ml/tools/llama.cpp/convert_hf_to_gguf.py
QUANT=/opt/llama.cpp/build/bin/llama-quantize
EXT=/srv/ml/longctx/gemma-4-26B-A4B-it-512k
OUT=/srv/ml/longctx/t87_llama
mkdir -p "$OUT"
echo "[$(date +%T)] convert ext -> F16"
"$PY" "$CONVERT" "$EXT" --outfile "$OUT/ext-f16.gguf" --outtype f16
echo "[$(date +%T)] quantize F16 -> Q6_K"
"$QUANT" "$OUT/ext-f16.gguf" "$OUT/ext-Q6_K.gguf" Q6_K
echo "[$(date +%T)] rm F16 intermediate"
rm -f "$OUT/ext-f16.gguf"
echo "[$(date +%T)] DONE: $(ls -la "$OUT/ext-Q6_K.gguf")"
