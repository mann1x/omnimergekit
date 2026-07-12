#!/usr/bin/env bash
# validate_imatrix_quality.sh — Phase B: QUALITY (HE+/MPE), gated on Phase A.
# Smoke (Phase A) shows termination; it CANNOT see a k-quant quality regression
# (k-quants don't ruminate). So the real "does the new imatrix hurt CD / beat v5"
# answer needs benchmark scores. This is the controlled quality comparison:
#
#   CD-Q4_K_M  (flagship k-quant CD tier, REGRESSION check):
#       shipped-imatrix  vs  winner-imatrix      -> must not drop
#   IQ3_XS     (i-quant RESCUE quality):
#       winner-imatrix only                       -> should score like a healthy
#                                                    tier (not just terminate)
#
# Internally controlled: we build + eval BOTH imatrices' CD-Q4_K_M here rather
# than trusting pre-recorded numbers (tokenizer/stack drift). Fixed CD map, so
# only the imatrix varies.
#
# Usage: validate_imatrix_quality.sh <winner-label>   (winner in {v5,bal,both};
#        default bal). Run AFTER validate_imatrix_corpora.sh and ONLY on free GPU0.
set -uo pipefail
WIN="${1:-bal}"
GPU=0
BIN=/opt/llama.cpp/build/bin
PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
TPL=/srv/ml/repos/omnimergekit/eval/templates
SCR=/srv/ml/repos/omnimergekit/scripts
TOK=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it
GD=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF
F16="$GD/gemma-4-A4B-98e-v7-coder-it-F16.gguf"
SHIPPED_IMAT="$GD/imatrix.dat"
WIN_IMAT="/mnt/sdc/ml/cd_fixed_v7/imat_matrix/imat_${WIN}.dat"
CD_MAP="$SCR/cd_maps_fixed_v7coder/tensor_types_CD-Q4_K_M.txt"
QST=/mnt/sdc/ml/cd_fixed_v7/imat_quality; mkdir -p "$QST"
RES=/srv/ml/eval_results_imat_quality
LOG=/srv/ml/logs/validate_imatrix_quality.txt; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1

for p in "$F16" "$SHIPPED_IMAT" "$WIN_IMAT" "$CD_MAP" "$OMK" "$TOK"; do
  [ -e "$p" ] || { echo "[FATAL] missing $p"; exit 1; }
done
echo "######## Phase B quality — winner=$WIN $(date -u) ########"

# build one GGUF: $1 outname  $2 imatrix  $3 tier  $4 cdmap(optional)
build() {
  local out="$QST/$1.gguf" lg="$QST/.q_$1.log"
  [ -f "$out" ] && { echo "  $1 exists"; return; }
  echo "  build $1 (tier=$3 imat=$(basename "$2")) $(date -u)"
  if [ -n "${4:-}" ]; then
    "$BIN/llama-quantize" --imatrix "$2" --tensor-type-file "$4" "$F16" "$out" "$3" >"$lg" 2>&1
  else
    "$BIN/llama-quantize" --imatrix "$2" "$F16" "$out" "$3" >"$lg" 2>&1
  fi
  [ -f "$out" ] || { echo "  [FATAL] build $1"; tail -4 "$lg"; exit 1; }
  echo "    $(du -h "$out"|cut -f1)"
}
echo "## build tiers (shipped+winner CD-Q4_K_M, winner IQ3_XS)"
build "ship_CD-Q4_K_M" "$SHIPPED_IMAT" Q4_K_M "$CD_MAP"
build "${WIN}_CD-Q4_K_M" "$WIN_IMAT" Q4_K_M "$CD_MAP"
build "${WIN}_IQ3_XS" "$WIN_IMAT" IQ3_XS ""

# eval one gguf on a bench: $1 served-name  $2 gguf  $3 template  $4 port
declare -A S
evalone() {
  local served="$1" gguf="$QST/$2.gguf" tplf="$TPL/$3.yaml" port="$4"
  echo "  eval $served / $3 (port $port) $(date -u)"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$OMK" --model "$gguf" --tokenizer "$TOK" \
    --template "$tplf" --backend llama --served-name "$served" \
    --results-dir "$RES" --port "$port" >"$QST/.eval_${served}_$3.log" 2>&1
  local sj; sj=$(find "$RES" -path "*$3*$served*summary.json" 2>/dev/null | head -1)
  local sc="ERR"; [ -n "$sj" ] && sc=$("$PY" -c "import json;print(json.load(open('$sj')).get('score'))" 2>/dev/null)
  S[$served/$3]="$sc"
  echo "    score=$sc"
}
port=8350
for cell in "ship_CD-Q4_K_M:ship_CD" "${WIN}_CD-Q4_K_M:${WIN}_CD" "${WIN}_IQ3_XS:${WIN}_IQ3XS"; do
  gguf="${cell%%:*}"; served="${cell##*:}"
  for bench in humanevalplus_full multipl_e_100; do
    evalone "$served" "$gguf" "$bench" "$port"; port=$((port+1))
  done
done

echo; echo "######## IMATRIX QUALITY (HE+ / MPE-100) — winner=$WIN ########"
printf "  %-22s | %-10s | %-10s\n" tier HE+ MPE-100
printf "  %-22s | %-10s | %-10s\n" "CD-Q4_K_M (shipped)" "${S[ship_CD/humanevalplus_full]:-?}" "${S[ship_CD/multipl_e_100]:-?}"
printf "  %-22s | %-10s | %-10s\n" "CD-Q4_K_M ($WIN)"      "${S[${WIN}_CD/humanevalplus_full]:-?}" "${S[${WIN}_CD/multipl_e_100]:-?}"
printf "  %-22s | %-10s | %-10s\n" "IQ3_XS ($WIN, rescued)" "${S[${WIN}_IQ3XS/humanevalplus_full]:-?}" "${S[${WIN}_IQ3XS/multipl_e_100]:-?}"
echo "  READ: CD-Q4_K_M shipped vs $WIN must be within noise (no regression);"
echo "        IQ3_XS ($WIN) should land near a healthy tier (Q4_K_M HE+ ~0.93) if truly rescued."
rm -f "$QST"/*.gguf
echo "[done] $(date -u)"
