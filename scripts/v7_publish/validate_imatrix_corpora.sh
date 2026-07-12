#!/usr/bin/env bash
# validate_imatrix_corpora.sh — Phase A: does a category-BALANCED imatrix corpus
# rescue the low i-quants WITHOUT regressing the quants that were already fine,
# does it beat the v5 corpus, and do we need BOTH?
#
# The imatrix is a GLOBAL input — swapping it changes EVERY imatrix-built quant
# (i-quants, CD tiers, imatrix k-quants), so we can't validate on i-quants alone.
# Matrix: 3 corpora x {coverage gate, IQ3_XS, IQ2_S, CD-Q4_K_M} on v7-coder 98e.
#   corpora: v5   = calibration_datav5.txt        (the existing/baseline corpus)
#            bal  = balanced_imatrix_v7.txt        (new, category-balanced)
#            both = v5 ++ bal                       ("need both" hypothesis)
#   tiers:   IQ3_XS, IQ2_S  -> i-quant RESCUE (were broken)
#            CD-Q4_K_M       -> REGRESSION check (was fine); FIXED CD map so only
#                               the imatrix varies (isolates imatrix-quality effect)
# Each imatrix is computed fresh from F16 with identical flags for a fair compare;
# the as-shipped imatrix.dat is also gated as a reference point.
# Quality (HE+/MPE) is a SEPARATE gated follow-up (validate_imatrix_quality.sh) on
# whichever corpus wins termination — smoke first, don't burn hours blind.
#
# Run ONLY when GPU0 is free (do not race the 2x2 / the GPU1 sweep).
set -uo pipefail
GPU=0
BIN=/opt/llama.cpp/build/bin
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
SMOKE=/srv/ml/scripts/smoke_gguf.sh
GATE="$SCR/imatrix_expert_coverage.py"
GD=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF
F16="$GD/gemma-4-A4B-98e-v7-coder-it-F16.gguf"
SHIPPED_IMAT="$GD/imatrix.dat"
CD_MAP="$SCR/cd_maps_fixed_v7coder/tensor_types_CD-Q4_K_M.txt"
COR_V5="$SCR/calibration_datav5.txt"
COR_BAL=/mnt/sdc/ml/corpora/balanced_imatrix_v7.txt
COR_BOTH=/mnt/sdc/ml/corpora/combined_v5_bal.txt
ST=/mnt/sdc/ml/cd_fixed_v7/imat_matrix; mkdir -p "$ST"
LOG=/srv/ml/logs/validate_imatrix_corpora.txt; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1

for p in "$F16" "$SHIPPED_IMAT" "$CD_MAP" "$COR_V5" "$COR_BAL" "$SMOKE" "$GATE"; do
  [ -e "$p" ] || { echo "[FATAL] missing $p"; exit 1; }
done

echo "######## build combined corpus (v5 ++ balanced) $(date -u) ########"
cat "$COR_V5" "$COR_BAL" > "$COR_BOTH"
echo "  v5=$(wc -l <"$COR_V5") lines  bal=$(wc -l <"$COR_BAL") lines  both=$(wc -l <"$COR_BOTH") lines"

declare -A IMAT
IMAT[v5]="$ST/imat_v5.dat"; IMAT[bal]="$ST/imat_bal.dat"; IMAT[both]="$ST/imat_both.dat"
declare -A COR
COR[v5]="$COR_V5"; COR[bal]="$COR_BAL"; COR[both]="$COR_BOTH"

echo; echo "######## 1. compute 3 imatrices (identical flags, GPU$GPU) ########"
for c in v5 bal both; do
  if [ -f "${IMAT[$c]}" ]; then echo "  imat_$c exists (skip)"; continue; fi
  echo "  -> imat_$c from ${COR[$c]} $(date -u)"
  CUDA_VISIBLE_DEVICES=$GPU "$BIN/llama-imatrix" -m "$F16" -f "${COR[$c]}" \
    -o "${IMAT[$c]}" -ngl 99 >"$ST/.imat_$c.log" 2>&1
  [ -f "${IMAT[$c]}" ] || { echo "  [FATAL] imat_$c build"; tail -6 "$ST/.imat_$c.log"; exit 1; }
  echo "     built $(du -h "${IMAT[$c]}"|cut -f1)"
done

echo; echo "######## 2. coverage gate — shipped + v5 + bal + both ########"
declare -A COV
gate_one() { # $1 label $2 imatrix
  local out; out=$("$PY" "$GATE" "$2" --json 2>/dev/null)
  local frac pass nstarve
  frac=$(echo "$out" | "$PY" -c "import json,sys;d=json.load(sys.stdin);print(f\"{d['worst_layer_starved_frac']:.3f}\")" 2>/dev/null)
  pass=$(echo "$out" | "$PY" -c "import json,sys;d=json.load(sys.stdin);print('PASS' if d['gate_pass'] else 'FAIL')" 2>/dev/null)
  nstarve=$(echo "$out" | "$PY" -c "import json,sys;d=json.load(sys.stdin);print(max(r['starved'] for r in d['layers']))" 2>/dev/null)
  COV[$1]="$pass worst_layer_frac=$frac max_starved=$nstarve/98"
  echo "  $1: ${COV[$1]}"
}
gate_one shipped "$SHIPPED_IMAT"
for c in v5 bal both; do gate_one "$c" "${IMAT[$c]}"; done

echo; echo "######## 3. termination matrix — {IQ3_XS, IQ2_S, CD-Q4_K_M} x {v5,bal,both} ########"
port=8330
declare -A R
build_smoke() { # $1 imat-label $2 tier
  local imat="${IMAT[$1]}" tier="$2" out="$ST/${1}_${2}.gguf" lg="$ST/.q_${1}_${2}.log"
  echo "==== build imat=$1 tier=$tier $(date -u) ===="
  if [ "$tier" = "CD-Q4_K_M" ]; then
    "$BIN/llama-quantize" --imatrix "$imat" --tensor-type-file "$CD_MAP" "$F16" "$out" Q4_K_M >"$lg" 2>&1
  else
    "$BIN/llama-quantize" --imatrix "$imat" "$F16" "$out" "$tier" >"$lg" 2>&1
  fi
  [ -f "$out" ] || { echo "  [FATAL] quant imat=$1 $tier"; tail -4 "$lg"; R[$1/$tier]=BUILD-FAIL; return; }
  local bpw; bpw=$(grep -oE "[0-9.]+ BPW" "$lg" | tail -1)
  echo "  built $(du -h "$out"|cut -f1) ($bpw)"
  local res; res=$(bash "$SMOKE" "$out" "$port" "$GPU" 2048 2>&1)
  echo "$res" | grep -E "RESULT:|STOP-ok|RUMINATE|BROKEN"
  R[$1/$tier]="$(echo "$res" | grep -oE "RESULT: [0-9]+/5 STOP" | head -1) ($bpw)"
  rm -f "$out"; port=$((port+1))
}
for c in v5 bal both; do
  for tier in IQ3_XS IQ2_S CD-Q4_K_M; do build_smoke "$c" "$tier"; done
done

echo; echo "######## IMATRIX-CORPUS VALIDATION MATRIX ########"
printf "  %-8s | %-34s | %-22s | %-22s | %-22s\n" corpus coverage-gate IQ3_XS IQ2_S CD-Q4_K_M
for c in v5 bal both; do
  printf "  %-8s | %-34s | %-22s | %-22s | %-22s\n" "$c" "${COV[$c]}" "${R[$c/IQ3_XS]:-?}" "${R[$c/IQ2_S]:-?}" "${R[$c/CD-Q4_K_M]:-?}"
done
echo "  (shipped imatrix coverage: ${COV[shipped]})"
echo
echo "  READ: rescue = i-quants flip to 5/5 STOP; regression = CD-Q4_K_M must stay 5/5 STOP;"
echo "        need-both = compare bal vs both coverage + i-quant STOP. Quality (HE+/MPE) is the"
echo "        next gated step on the winning corpus (validate_imatrix_quality.sh)."
echo "[done] $(date -u)"
