#!/usr/bin/env bash
# build_std16_cd_q2k.sh — cut STD16 (v7-coder force-keep promotion candidate) CD-Q2_K from the
# STD16 regular F16 + cohort imatrix + the FIXED v7 CD tensor-type-file. The CD overrides are
# per-tensor-NAME (protect attn_v/k + ffn_down x30 at K-quant — the #563 rumination fix), so they
# transfer across drop maps. CPU-only quantize: no GPU contention with the running plain loop gate.
# Output lands in the cohort GGUF dir; loop-gate is launched separately when a GPU frees.
set -uo pipefail
BIN=/opt/llama.cpp/build/bin
GG=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF
F16="$GG/gemma-4-A4B-98e-v7-coder-it-F16.gguf"
IMAT="$GG/imatrix.dat"
CD=/srv/ml/scripts/cd_maps_v7_fixed/coder/tensor_types_CD-Q2_K.txt
OUT="$GG/gemma-4-A4B-98e-v7-coder-it-CD-Q2_K.gguf"
LOG=/mnt/sdc/ml/t223_fk/build_std16_cd_q2k.log
exec > >(tee -a "$LOG") 2>&1
ts(){ date -u +%T; }
echo "==================== STD16 CD-Q2_K build $(ts) UTC ===================="
for f in "$F16" "$IMAT" "$CD" "$BIN/llama-quantize"; do [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done
[ "$(grep -cE '=IQ' "$CD")" -eq 0 ] || { echo "FATAL CD map has i-quant slots (want k-quant-floor body)"; exit 1; }
echo "F16=$(du -h "$F16"|cut -f1)  imat=$(du -h "$IMAT"|cut -f1)  cd overrides=$(grep -cE '=' "$CD")"
magic(){ head -c4 "$1" 2>/dev/null; }
if [ -s "$OUT" ] && [ "$(magic "$OUT")" = GGUF ]; then
  echo "CD-Q2_K already built, skip"
else
  "$BIN/llama-quantize" --imatrix "$IMAT" --tensor-type-file "$CD" "$F16" "$OUT" Q2_K
fi
[ "$(magic "$OUT")" = GGUF ] || { echo "FATAL build failed (bad magic)"; exit 1; }
echo "BUILT CD-Q2_K = $(du -h "$OUT"|cut -f1)  $(grep -oE '[0-9.]+ BPW' "$LOG" | tail -1)"
echo "STD16_CD_Q2K_BUILD_DONE"
