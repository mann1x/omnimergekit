#!/usr/bin/env bash
# verify_safe_cd.sh — build + termination-smoke the 2 "safe-by-design" published
# v7 CD tiers (CD-Q6_K low=Q5_K, CD-Q5_K_M low=Q4_K) for both models, using the
# already-regenerated maps. These were never IQ3_S so should STOP — but the whole
# lesson is "verify, don't assume." Staging only. GPU0.
set -uo pipefail
GPU=0
QUANT=/opt/llama.cpp/build/bin/llama-quantize
SMOKE=/srv/ml/scripts/smoke_gguf.sh
STAGE=/mnt/sdc/ml/cd_fixed_v7
MAPROOT=/srv/ml/scripts/cd_maps_v7_fixed
LOG=/srv/ml/logs/verify_safe_cd.txt
: > "$LOG"; exec > >(tee -a "$LOG") 2>&1
declare -A F16 IMAT
F16[coder]=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF/gemma-4-A4B-98e-v7-coder-it-F16.gguf
IMAT[coder]=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF/imatrix.dat
F16[coderx]=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it-GGUF/gemma-4-A4B-98e-v7-coderx-it-F16.gguf
IMAT[coderx]=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it-GGUF/imatrix.dat
TIERS=(CD-Q6_K CD-Q5_K_M)
port=8280
declare -A V
for m in coder coderx; do
  for t in "${TIERS[@]}"; do
    mp="$MAPROOT/$m/tensor_types_${t}.txt"; ft="${t#CD-}"
    out="$STAGE/gemma-4-A4B-98e-v7-${m}-it-${t}.gguf"
    [ -f "$mp" ] || { echo "[FATAL] missing map $mp"; exit 1; }
    echo "==== build $m $t (ftype=$ft) $(date -u) ===="
    "$QUANT" --imatrix "${IMAT[$m]}" --tensor-type-file "$mp" "${F16[$m]}" "$out" "$ft" >"$STAGE/.q_${m}_${t}.log" 2>&1
    [ $? -eq 0 ] && [ -f "$out" ] || { echo "[FATAL] quant $m $t"; tail -3 "$STAGE/.q_${m}_${t}.log"; V[$m/$t]=BUILD-FAIL; continue; }
    echo "  built $(du -h "$out"|cut -f1) -> $out"
    res=$(bash "$SMOKE" "$out" "$port" "$GPU" 2048 2>&1)
    echo "$res" | grep -E "STOP-ok|RUMINATE|500-ERR|RESULT:"
    V[$m/$t]=$(echo "$res" | grep -oE "RESULT: [0-9]+/5 STOP" | head -1)
    port=$((port+1))
  done
done
echo "######## SAFE-TIER GATE SUMMARY ########"
for k in "${!V[@]}"; do printf "  %-22s %s\n" "$k" "${V[$k]}"; done | sort
echo "[done] $(date -u)"
