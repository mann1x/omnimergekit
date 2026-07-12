#!/usr/bin/env bash
# build_add16_q3ks.sh — cut ADD16-Q3_K_S from the on-disk ADD16 bf16 (ADD16-combo) using the
# EXACT cohort recipe (llama.cpp-latest convert --outtype f16 -> llama-quantize --imatrix … Q3_K_S),
# so it is apples-to-apples with the STD16 cohort tiers the gate is testing. ADD16's own imatrix
# (preserved) is reused. CPU-only. The F16 intermediate is KEPT (we may cut more ADD16 tiers).
set -uo pipefail
LCPP=/mnt/sdc/ml/llama.cpp-latest
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SRC=/mnt/sdc/ml/t223_fk/ADD16-combo
F16=/mnt/sdc/ml/t223_fk/ADD16-F16.gguf
IMAT=/mnt/sdc/ml/t223_fk/ADD16-imatrix.dat
OUT=/mnt/sdc/ml/t223_fk/ADD16-Q3_K_S.gguf
ts(){ date '+%T %Z'; }

[ -f "$SRC/model.safetensors" ] || { echo "FATAL: no ADD16 bf16 at $SRC"; exit 2; }
[ -f "$IMAT" ] || { echo "FATAL: no ADD16 imatrix at $IMAT"; exit 2; }
[ -x "$LCPP/build/bin/llama-quantize" ] || { echo "FATAL: no llama-quantize at $LCPP"; exit 2; }

if [ ! -f "$F16" ]; then
  echo "[$(ts)] convert ADD16 bf16 -> F16"
  "$PY" "$LCPP/convert_hf_to_gguf.py" "$SRC" --outfile "$F16" --outtype f16 \
    || { echo "CONVERT_FAIL"; exit 1; }
else
  echo "[$(ts)] F16 already present, reusing $F16"
fi
echo "[$(ts)] F16: $(stat -c%s "$F16") bytes"

echo "[$(ts)] quantize Q3_K_S (imatrix=$IMAT)"
"$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$OUT" Q3_K_S 32 \
  || { echo "QUANT_FAIL"; exit 1; }
echo "[$(ts)] DONE: $(stat -c%s "$OUT") bytes -> $OUT"
echo "ADD16_Q3KS_BUILD_DONE"
