#!/usr/bin/env bash
# build_128e_q3ks_baseref.sh <GPU> — cut the UNPRUNED 128e A4B at Q3_K_S as the loop base
# reference. Matched to STD16/ADD16: same llama.cpp-latest, same calib_both.txt imatrix recipe,
# same Q3_K_S tier — so "does the base loop at Q3_K_S?" isolates quant-induced from prune-induced
# looping. imatrix needs a GPU (~15 min, 128 chunks ngl99); quant is CPU. imatrix.dat is PRESERVED.
set -uo pipefail
GPU="${1:?usage: build_128e_q3ks_baseref.sh <GPU_ID>}"
LCPP=/mnt/sdc/ml/llama.cpp-latest
F16=/mnt/sdc/ml/eval_gguf/128e-F16.gguf
CALIB=/mnt/sdc/ml/qat_investig/calib_both.txt          # SAME calib as STD16/ADD16 imatrix
OUTDIR=/mnt/sdc/ml/t223_fk
IMAT=$OUTDIR/128e-imatrix.dat
OUT=$OUTDIR/128e-imat-Q3_K_S.gguf
LOG=$OUTDIR/128e_q3ks_build.log
ts(){ date '+%T %Z'; }

[ -f "$F16" ]   || { echo "FATAL: no 128e F16 at $F16"; exit 2; }
[ -f "$CALIB" ] || { echo "FATAL: no calib at $CALIB"; exit 2; }
[ -x "$LCPP/build/bin/llama-imatrix" ]  || { echo "FATAL: no llama-imatrix"; exit 2; }
[ -x "$LCPP/build/bin/llama-quantize" ] || { echo "FATAL: no llama-quantize"; exit 2; }

if [ ! -f "$IMAT" ]; then
  echo "[$(ts)] 128e imatrix on GPU$GPU (calib_both 128 chunks ngl99) -> $IMAT"
  CUDA_VISIBLE_DEVICES="$GPU" "$LCPP/build/bin/llama-imatrix" \
    -m "$F16" -f "$CALIB" -o "$IMAT" --chunks 128 -ngl 99 \
    || { echo "IMATRIX_FAIL"; exit 1; }
else
  echo "[$(ts)] 128e imatrix already present, reusing $IMAT"
fi
echo "[$(ts)] imatrix: $(stat -c%s "$IMAT") bytes (PRESERVED)"

echo "[$(ts)] quantize 128e Q3_K_S (imatrix matched)"
"$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$OUT" Q3_K_S 32 \
  || { echo "QUANT_FAIL"; exit 1; }
echo "[$(ts)] DONE: $(stat -c%s "$OUT") bytes -> $OUT"
echo "BASE128E_Q3KS_BUILD_DONE"
