#!/usr/bin/env bash
# smoke_published_iquants.sh — termination-gate the two PUBLISHED low i-quants
# flagged as rumination-suspect (IQ3_M 3.66bpw, IQ2_XS 2.06bpw) on both v7 models,
# plus their clean k-quant neighbors (Q3_K_S, Q2_K) as replacement candidates.
# Build from F16 (+imatrix) on CPU, smoke on GPU0. Staging only.
set -uo pipefail
GPU=0
QUANT=/opt/llama.cpp/build/bin/llama-quantize
SMOKE=/srv/ml/scripts/smoke_gguf.sh
STAGE=/mnt/sdc/ml/cd_fixed_v7/iquant_probe
mkdir -p "$STAGE"
LOG=/srv/ml/logs/smoke_published_iquants.txt
: > "$LOG"; exec > >(tee -a "$LOG") 2>&1
declare -A F16 IMAT
F16[coder]=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF/gemma-4-A4B-98e-v7-coder-it-F16.gguf
IMAT[coder]=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF/imatrix.dat
F16[coderx]=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it-GGUF/gemma-4-A4B-98e-v7-coderx-it-F16.gguf
IMAT[coderx]=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it-GGUF/imatrix.dat
# published suspects + k-quant replacement candidates
TIERS=(IQ3_M IQ2_XS Q3_K_S Q2_K)
port=8290
declare -A V
for m in coder coderx; do
  for t in "${TIERS[@]}"; do
    out="$STAGE/gemma-4-A4B-98e-v7-${m}-it-${t}.gguf"
    echo "==== build $m $t $(date -u) ===="
    "$QUANT" --imatrix "${IMAT[$m]}" "${F16[$m]}" "$out" "$t" >"$STAGE/.q_${m}_${t}.log" 2>&1
    [ $? -eq 0 ] && [ -f "$out" ] || { echo "[FATAL] quant $m $t"; tail -3 "$STAGE/.q_${m}_${t}.log"; V[$m/$t]=BUILD-FAIL; continue; }
    echo "  built $(du -h "$out"|cut -f1)"
    res=$(bash "$SMOKE" "$out" "$port" "$GPU" 2048 2>&1)
    echo "$res" | grep -E "STOP-ok|RUMINATE|500-ERR|RESULT:"
    V[$m/$t]=$(echo "$res" | grep -oE "RESULT: [0-9]+/5 STOP" | head -1)
    rm -f "$out"   # probe only — don't keep
    port=$((port+1))
  done
done
echo "######## PUBLISHED I-QUANT GATE SUMMARY ########"
for k in "${!V[@]}"; do printf "  %-20s %s\n" "$k" "${V[$k]}"; done | sort
echo "[done] $(date -u)"
