#!/usr/bin/env bash
# regen_build_smoke_cd.sh — regenerate v7 CD maps with the patched generator
# (low-tier i-quant -> k-quant) for BOTH models from their own imatrix, then
# rebuild + termination-smoke the 3 in-scope tiers each. Staging output only;
# nothing is promoted/published. GPU0 (dedicated CD GPU). Runs sequentially.
set -uo pipefail
GPU=0
PY=/srv/ml/envs/envs/omnimergekit/bin/python
GEN=/srv/ml/repos/omnimergekit/scripts/generate_cd_maps.py
QUANT=/opt/llama.cpp/build/bin/llama-quantize
SMOKE=/srv/ml/scripts/smoke_gguf.sh
STAGE=/mnt/sdc/ml/cd_fixed_v7
MAPROOT=/srv/ml/scripts/cd_maps_v7_fixed
LOG=/srv/ml/logs/regen_build_smoke_cd.txt
mkdir -p "$STAGE" "$MAPROOT"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

# model_key : F16 path : imatrix path : output stem
declare -A F16 IMAT
F16[coder]=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF/gemma-4-A4B-98e-v7-coder-it-F16.gguf
IMAT[coder]=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF/imatrix.dat
F16[coderx]=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it-GGUF/gemma-4-A4B-98e-v7-coderx-it-F16.gguf
IMAT[coderx]=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it-GGUF/imatrix.dat

TIERS=(CD-Q4_K_M CD-Q3_K_L CD-IQ4_K_M)
# file-base FTYPE: strip CD- prefix, except CD-IQ4_K_M -> Q5_K (canonical override)
ftype_of() { case "$1" in CD-IQ4_K_M) echo Q5_K;; *) echo "${1#CD-}";; esac; }

port=8270
declare -A VERDICT
for m in coder coderx; do
  f16="${F16[$m]}"; imat="${IMAT[$m]}"; mapdir="$MAPROOT/$m"
  for f in "$f16" "$imat"; do [ -f "$f" ] || { echo "[FATAL] missing $f"; exit 1; }; done
  echo "######## $m : regen maps from $(basename "$imat") ########"
  mkdir -p "$mapdir"
  "$PY" "$GEN" --imatrix "$imat" --out-dir "$mapdir" || { echo "[FATAL] map regen $m"; exit 1; }
  # confirm the fix actually landed in the maps (no IQ3_S/IQ2_S in the 3 tiers)
  for t in "${TIERS[@]}"; do
    mp="$mapdir/tensor_types_${t}.txt"
    bad=$(grep -cE "=IQ3_S|=IQ2_S" "$mp" 2>/dev/null || echo 0)
    echo "  map $t: $(wc -l <"$mp") lines, low-iquant lines=$bad, Q3_K=$(grep -c '=Q3_K' "$mp")"
  done
  for t in "${TIERS[@]}"; do
    mp="$mapdir/tensor_types_${t}.txt"
    ft=$(ftype_of "$t")
    out="$STAGE/gemma-4-A4B-98e-v7-${m}-it-${t}.gguf"
    echo "==== build $m $t (ftype=$ft) $(date -u) ===="
    "$QUANT" --imatrix "$imat" --tensor-type-file "$mp" "$f16" "$out" "$ft" >"$STAGE/.q_${m}_${t}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$out" ]; then echo "[FATAL] quant $m $t rc=$rc"; tail -3 "$STAGE/.q_${m}_${t}.log"; VERDICT[$m/$t]="BUILD-FAIL"; continue; fi
    echo "  built $(du -h "$out" | cut -f1) -> $out"
    echo "==== smoke $m $t on GPU$GPU port$port ===="
    res=$(bash "$SMOKE" "$out" "$port" "$GPU" 2048 2>&1)
    echo "$res" | grep -E "STOP-ok|RUMINATE|500-ERR|PARSE-ERR|RESULT:"
    VERDICT[$m/$t]=$(echo "$res" | grep -oE "RESULT: [0-9]+/5 STOP" | head -1)
    port=$((port+1))
  done
done

echo "######## SMOKE-GATE SUMMARY ########"
for k in "${!VERDICT[@]}"; do printf "  %-22s %s\n" "$k" "${VERDICT[$k]}"; done | sort
echo "[done] $(date -u)"
