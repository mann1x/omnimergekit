#!/usr/bin/env bash
# imat_promote.sh <bf16_dir> <label> <gpu> <port>
# Promote a noimat DERN variant to imat-Q6: F16 -> model-specific imatrix -> imat-Q6 -> loop gate.
# imatrix calib is IDENTICAL to dern11's (calib_both.txt, 128 chunks, -ngl 99) so the result is
# apples-to-apples vs dern11-imat-Q6 (1/48@t0.9, 2/48@t0.8). imatrix.dat is PRESERVED (mandatory).
set -uo pipefail
BF16=$1; LABEL=$2; GPU=$3; PORT=$4
PY=/root/anaconda3/envs/omnimergekit/bin/python
LCPP=/mnt/sdc/ml/llama.cpp-latest
SFT=/mnt/sdc/ml/sft_heal
CALIB=/mnt/sdc/ml/qat_investig/calib_both.txt
GATE=/srv/ml/scripts/gate_sweep48_minp_p.sh
AL=/srv/ml/agentic_loop
F16=$SFT/imatpromote-$LABEL-F16.gguf
IMAT=$SFT/$LABEL-imatrix.dat
Q6=$SFT/gemma-4-A4B-98e-v7-coder-$LABEL-imat-Q6_K.gguf
RESULT=$AL/results/${LABEL}-imatQ6_minp48.json
ts(){ date '+%T %Z'; }
echo "[imat-$LABEL $(ts)] bf16=$(basename "$BF16") GPU=$GPU calib=$(basename "$CALIB")"
for f in "$BF16/config.json" "$CALIB" "$GATE" "$LCPP/build/bin/llama-imatrix"; do
  [ -e "$f" ] || { echo "[$LABEL] FATAL missing $f"; exit 9; }
done
# 1. F16 (reconvert from bf16; fx-2.0 F16 was cleaned)
[ -f "$F16" ] || "$PY" "$LCPP/convert_hf_to_gguf.py" "$BF16" --outfile "$F16" --outtype f16 \
  || { echo "[$LABEL] FATAL convert"; exit 2; }
# 2. model-specific imatrix (same calib/chunks/ngl as dern11) -> PRESERVED
if [ ! -f "$IMAT" ]; then
  CUDA_VISIBLE_DEVICES=$GPU "$LCPP/build/bin/llama-imatrix" \
    -m "$F16" -f "$CALIB" -o "$IMAT" --chunks 128 -ngl 99 \
    || { echo "[$LABEL] FATAL imatrix"; exit 4; }
fi
# 3. imat-Q6
[ -f "$Q6" ] || "$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$Q6" Q6_K 32 \
  || { echo "[$LABEL] FATAL quant"; exit 3; }
# 4. loop gate vs dern11-imat-Q6 (1/48,2/48)
echo "[imat-$LABEL $(ts)] gate {0.9,0.8} GPU$GPU:$PORT vs dern11-imat-Q6 1/48,2/48"
bash "$GATE" "$Q6" "$GPU" "$PORT" "$RESULT" "$LABEL-imatQ6"
# 5. drop F16 only (imatrix.dat + Q6 kept)
rm -f "$F16"
echo "[imat-$LABEL $(ts)] DONE  Q6=$Q6  imatrix=$IMAT"
