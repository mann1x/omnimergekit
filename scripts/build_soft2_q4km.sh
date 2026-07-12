#!/usr/bin/env bash
# build_soft2_q4km.sh — build the DERN soft2 (loop-fix) Q4_K_M imatrix GGUF.
# CPU-only (convert + quantize); no GPU. Reuses the PRESERVED model-specific
# imatrix (dern11-soft-soft2-imatrix.dat) so it is internally consistent with
# the soft2 imat-Q6 ship candidate. F16 intermediate is deleted after quant.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
LCPP=/mnt/sdc/ml/llama.cpp-latest
SFT=/mnt/sdc/ml/sft_heal
BF16=$SFT/dern11-soft-soft2-it
IMAT=$SFT/dern11-soft-soft2-imatrix.dat
F16=$SFT/soft2-q4km-F16.gguf
Q4=$SFT/gemma-4-A4B-98e-v7-coder-dern11-soft-soft2-imat-Q4_K_M.gguf
ts(){ date '+%T %Z'; }
echo "[q4km $(ts)] start  bf16=$(basename "$BF16")  imat=$(basename "$IMAT")"
for f in "$BF16/model.safetensors" "$IMAT" "$LCPP/convert_hf_to_gguf.py" "$LCPP/build/bin/llama-quantize"; do
  [ -e "$f" ] || { echo "[q4km] FATAL missing $f"; exit 9; }
done
# 1. convert bf16 -> F16
[ -f "$F16" ] || "$PY" "$LCPP/convert_hf_to_gguf.py" "$BF16" --outfile "$F16" --outtype f16 \
  || { echo "[q4km] FATAL convert"; exit 2; }
# 2. quantize Q4_K_M with the preserved imatrix
[ -f "$Q4" ] || "$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$Q4" Q4_K_M 32 \
  || { echo "[q4km] FATAL quant"; exit 3; }
# 3. magic-header sanity
magic=$("$PY" -c "import sys;print(open('$Q4','rb').read(4).decode('latin1'))" 2>/dev/null)
[ "$magic" = "GGUF" ] || { echo "[q4km] FATAL bad GGUF header"; exit 4; }
# 4. drop F16 (Q4_K_M + imatrix kept)
rm -f "$F16"
ls -la "$Q4"
echo "[q4km $(ts)] Q4KM_DONE  $Q4"
