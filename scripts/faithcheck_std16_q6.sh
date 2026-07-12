#!/usr/bin/env bash
# faithcheck_std16_q6.sh — prove the rebuilt STD16 F16 is byte-faithful to the 8/9-eval'd model.
# Re-quant rebuilt F16 -> imat-Q6 with the existing STD16-imatrix.dat, SHA-compare to the
# eval'd STD16-imatq6.gguf. Q6_K+imatrix is deterministic (thread-count independent), so a
# byte match proves the rebuilt F16 == original F16 -> the whole cohort built from it is trusted.
set -uo pipefail
LCPP=/mnt/sdc/ml/llama.cpp-latest
WORK=/mnt/sdc/ml/t223_fk
F16=$WORK/STD16-F16.gguf
IMAT=$WORK/STD16-imatrix.dat
REF=$WORK/STD16-imatq6.gguf
TMP=$WORK/STD16-imatq6-rebuilt.gguf
ts(){ date '+%T %Z'; }

for f in "$LCPP/build/bin/llama-quantize" "$F16" "$IMAT" "$REF"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
echo "[faith $(ts)] re-quant rebuilt F16 -> Q6_K (imatrix=$IMAT)"
"$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$TMP" Q6_K 32 \
  > "$WORK/faithcheck_quant.log" 2>&1 || { echo "FATAL quant (see faithcheck_quant.log)"; tail -5 "$WORK/faithcheck_quant.log"; exit 1; }

echo "[faith $(ts)] sha256 compare"
a=$(sha256sum "$TMP" | awk '{print $1}'); sa=$(stat -c%s "$TMP")
b=$(sha256sum "$REF" | awk '{print $1}'); sb=$(stat -c%s "$REF")
echo "rebuilt: $a  size=$sa"
echo "evald  : $b  size=$sb"
if [ "$a" = "$b" ]; then
  rm -f "$TMP"
  echo "[faith $(ts)] FAITHFUL_MATCH — rebuilt F16 is byte-identical to the eval'd STD16; temp removed, $REF kept as Q6 tier"
  echo "STD16_FAITHCHECK_PASS"
else
  echo "[faith $(ts)] FAITHFUL_MISMATCH — sizes ref=$sb rebuilt=$sa; keeping both for diff"
  echo "STD16_FAITHCHECK_FAIL"
fi
