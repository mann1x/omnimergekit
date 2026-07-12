#!/usr/bin/env bash
# validate_balanced_imatrix.sh — does a category-BALANCED imatrix corpus close
# the per-(expert,layer) coverage gap and stop the i-quant rumination?
#
# Proof, end-to-end, on v7-coder 98e:
#   1. coverage gate on the OLD (code-heavy) imatrix         -> before
#   2. compute a NEW imatrix from the balanced corpus        -> GPU0, ~15-20 min
#   3. coverage gate on the NEW imatrix                      -> after (0 starved?)
#   4. rebuild IQ3_XS + IQ2_S from the NEW imatrix + smoke   -> do they STOP now?
# If the NEW-imatrix i-quants STOP, the corpus rescue is proven and we wire the
# balanced corpus + coverage gate into quantize_gguf.py as the canonical path.
#
# Run ONLY when GPU0 is free (do not race the 2x2 / the GPU1 sweep).
set -uo pipefail
GPU=0
BIN=/opt/llama.cpp/build/bin
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
SMOKE=/srv/ml/scripts/smoke_gguf.sh
GD=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF
F16="$GD/gemma-4-A4B-98e-v7-coder-it-F16.gguf"
OLD_IMAT="$GD/imatrix.dat"
CORPUS=/mnt/sdc/ml/corpora/balanced_imatrix_v7.txt
ST=/mnt/sdc/ml/cd_fixed_v7/imat_rescue; mkdir -p "$ST"
NEW_IMAT="$ST/imatrix_balanced.dat"
LOG=/srv/ml/logs/validate_balanced_imatrix.txt; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1

for p in "$F16" "$OLD_IMAT" "$CORPUS" "$SMOKE" "$SCR/imatrix_expert_coverage.py"; do
  [ -e "$p" ] || { echo "[FATAL] missing $p"; exit 1; }
done

echo "######## 1. coverage gate — OLD (code-heavy) imatrix $(date -u) ########"
"$PY" "$SCR/imatrix_expert_coverage.py" "$OLD_IMAT" --top 6 || true

echo; echo "######## 2. compute NEW imatrix from balanced corpus (GPU$GPU) ########"
if [ -f "$NEW_IMAT" ]; then
  echo "  NEW imatrix already present: $NEW_IMAT (skip compute)"
else
  CUDA_VISIBLE_DEVICES=$GPU "$BIN/llama-imatrix" -m "$F16" -f "$CORPUS" \
    -o "$NEW_IMAT" -ngl 99 --no-warmup >"$ST/.imatrix_build.log" 2>&1
  if [ $? -ne 0 ] || [ ! -f "$NEW_IMAT" ]; then echo "[FATAL] imatrix build"; tail -8 "$ST/.imatrix_build.log"; exit 1; fi
  echo "  built $(du -h "$NEW_IMAT"|cut -f1)"
fi

echo; echo "######## 3. coverage gate — NEW (balanced) imatrix ########"
"$PY" "$SCR/imatrix_expert_coverage.py" "$NEW_IMAT" --top 6
gate=$?
echo "  coverage gate exit=$gate (0=PASS, 2=still starved)"

echo; echo "######## 4. rebuild IQ3_XS + IQ2_S from NEW imatrix + smoke ########"
port=8320
declare -A V
for tier in IQ3_XS IQ2_S; do
  out="$ST/${tier}_balanced.gguf"
  echo "==== build $tier (balanced imatrix) $(date -u) ===="
  "$BIN/llama-quantize" --imatrix "$NEW_IMAT" "$F16" "$out" "$tier" >"$ST/.q_${tier}.log" 2>&1
  [ $? -eq 0 ] && [ -f "$out" ] || { echo "[FATAL] quant $tier"; tail -4 "$ST/.q_${tier}.log"; V[$tier]=BUILD-FAIL; continue; }
  bpw=$(grep -oE "[0-9.]+ BPW" "$ST/.q_${tier}.log" | tail -1)
  echo "  built $(du -h "$out"|cut -f1) ($bpw)"
  res=$(bash "$SMOKE" "$out" "$port" "$GPU" 2048 2>&1)
  echo "$res" | grep -E "RESULT:|STOP-ok|RUMINATE|BROKEN"
  V[$tier]="$(echo "$res" | grep -oE "RESULT: [0-9]+/5 STOP" | head -1) ($bpw)"
  rm -f "$out"
  port=$((port+1))
done

echo; echo "######## BALANCED-IMATRIX RESCUE SUMMARY ########"
echo "  old-imatrix i-quants ruminate (5/5 BROKEN, established)."
for k in "${!V[@]}"; do printf "  %-10s NEW-imatrix: %s\n" "$k" "${V[$k]}"; done | sort
echo "  VERDICT: if NEW-imatrix IQ3_XS/IQ2_S show >=4/5 STOP, the corpus rescue works."
echo "[done] $(date -u)"
