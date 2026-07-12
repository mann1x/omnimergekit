#!/usr/bin/env bash
# build_std16_cd_iq2nl.sh — build the 2 CD-IQ2_NL variants for the STD16 cohort from the STD16 F16
# + cohort imatrix + the blend-ranked CD-IQ2_NL tensor-type maps. This IS a deliberate i-quant tier:
#   bulk IQ3_S, tail IQ2_S (CD-IQ2_NL) / Q2_K (CD-IQ2_NL-q2k), attn+router IQ4_NL, ffn_down IQ3_S floor.
# Base ftype IQ3_S (un-overridden tensors). CPU-only quantize (no GPU contention). Output -> cohort
# GGUF dir; both loop-gated separately. imatrix is MANDATORY for i-quants (already preserved in dir).
set -uo pipefail
BIN=/opt/llama.cpp/build/bin
GG=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF
STEM=gemma-4-A4B-98e-v7-coder-it
F16="$GG/$STEM-F16.gguf"
IMAT="$GG/imatrix.dat"
MAPDIR=/srv/ml/scripts/cd_maps_std16_iq2nl
WORK=/mnt/sdc/ml/t223_fk
LOG="$WORK/build_std16_cd_iq2nl.log"
exec > >(tee -a "$LOG") 2>&1
ts(){ date -u +%T; }
magic(){ head -c4 "$1" 2>/dev/null; }
echo "==================== STD16 CD-IQ2_NL build $(ts) UTC ===================="
for f in "$F16" "$IMAT" "$BIN/llama-quantize"; do [ -e "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

build_one(){ # name  base_ftype
  local NAME="$1" BASE="$2"
  local CD="$MAPDIR/tensor_types_${NAME}.txt"
  local OUT="$GG/$STEM-${NAME}.gguf"
  [ -f "$CD" ] || { echo "FATAL missing map $CD"; return 1; }
  # deliberate i-quant tier: ASSERT i-quant slots PRESENT (inverse of the K-floor guard)
  [ "$(grep -cE '=IQ' "$CD")" -gt 0 ] || { echo "FATAL $NAME map has NO i-quant slots"; return 1; }
  if [ -s "$OUT" ] && [ "$(magic "$OUT")" = GGUF ]; then echo "[$(ts)] $NAME exists, skip"; else
    echo "[$(ts)] llama-quantize $NAME (base $BASE, imatrix + tensor-type-file)"
    "$BIN/llama-quantize" --imatrix "$IMAT" --tensor-type-file "$CD" "$F16" "$OUT" "$BASE" 2>&1 | tail -5
  fi
  [ "$(magic "$OUT")" = GGUF ] || { echo "FATAL $NAME bad magic header"; return 1; }
  local sz szb; sz=$(du -h "$OUT"|cut -f1); szb=$(stat -c%s "$OUT")
  echo "[$(ts)] BUILT $NAME = $sz"
  # i-quant fallback to F16 would balloon size; flag if > 13 GB (expected ~8)
  [ "$szb" -gt 13000000000 ] && echo "[$(ts)] WARN $NAME unexpectedly large ($sz) — check for i-quant fallback"
  return 0
}

build_one CD-IQ2_NL     IQ3_S
build_one CD-IQ2_NL-q2k IQ3_S
echo "[$(ts)] ==================== CD-IQ2_NL build DONE ===================="
ls -la "$GG"/${STEM}-CD-IQ2_NL*.gguf 2>/dev/null
echo "STD16_CD_IQ2NL_BUILD_DONE"
