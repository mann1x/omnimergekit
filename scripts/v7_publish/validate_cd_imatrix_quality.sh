#!/usr/bin/env bash
# validate_cd_imatrix_quality.sh — CORRECTED Phase B (supersedes validate_imatrix_quality.sh).
#
# The original Phase A/B used the WRONG CD map (cd_maps_fixed_v7coder/tensor_types_CD-Q4_K_M.txt
# has 176 IQ3_S i-quant slots -> ruminates by construction -> bogus 5/5 BROKEN "regression").
# The real shipped CD tier uses the K-QUANT-FLOOR map (176 Q3_K, zero i-quant slots), validated
# 5/5 STOP on both coder+coderx (task #564). Termination is therefore NOT in question here.
#
# Open question this answers: does the bal/both imatrix improve CD-Q4_K_M *quality* (HE+/MPE)
# over the v5 imatrix, using the correct k-quant-floor map? Fixed map, only the imatrix varies.
#   - v5   imatrix (calibration_datav5.txt, 421k tok)        -> baseline
#   - both imatrix (v5 ++ balanced, 573k tok, best coverage) -> candidate new default
# Reuses the prebuilt imat_v5.dat / imat_both.dat (imatrix compute was valid; only the CD map
# and the volume-confounded coverage gate were wrong). Greedy, llama backend, GPU0 only.
#
# NOTE: coverage gate intentionally NOT used — it is volume-confounded (relative <20%-of-mean
# threshold penalises smaller corpora) and was anti-predictive of actual i-quant termination.
set -uo pipefail
# omk_eval shells out to the `lm-eval` console script; a detached setsid/nohup launch
# inherits a minimal PATH without the conda env bin -> FileNotFoundError: 'lm-eval'.
# Put the omk env bin first so lm-eval (and python) resolve.
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH
GPU=0
BIN=/opt/llama.cpp/build/bin
PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
TPL=/srv/ml/repos/omnimergekit/eval/templates
SCR=/srv/ml/repos/omnimergekit/scripts
SMOKE=/srv/ml/scripts/smoke_gguf.sh
TOK=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it
GD=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF
F16="$GD/gemma-4-A4B-98e-v7-coder-it-F16.gguf"
CD_MAP="$SCR/cd_maps_v7_fixed/coder/tensor_types_CD-Q4_K_M.txt"   # k-quant floor: 176 Q3_K, 0 IQ
IMAT_DIR=/mnt/sdc/ml/cd_fixed_v7/imat_matrix
declare -A IMAT; IMAT[v5]="$IMAT_DIR/imat_v5.dat"; IMAT[both]="$IMAT_DIR/imat_both.dat"
QST=/mnt/sdc/ml/cd_fixed_v7/imat_quality; mkdir -p "$QST"
RES=/srv/ml/eval_results_imat_quality
LOG=/srv/ml/logs/validate_cd_imatrix_quality.txt
: > "$LOG"; exec >>"$LOG" 2>&1   # single redirect (NO tee) — avoids the Phase-A double-write

for p in "$F16" "$CD_MAP" "$OMK" "$TOK" "$SMOKE" "${IMAT[v5]}" "${IMAT[both]}"; do
  [ -e "$p" ] || { echo "[FATAL] missing $p"; exit 1; }
done
nIQ=$(grep -cE "=IQ" "$CD_MAP")
[ "$nIQ" -eq 0 ] || { echo "[FATAL] CD_MAP has $nIQ i-quant slots — wrong map ($CD_MAP)"; exit 1; }
echo "######## corrected Phase B — {v5,both} x CD-Q4_K_M (k-quant floor) $(date -u) ########"
echo "  CD_MAP=$CD_MAP  (i-quant slots=$nIQ, expect 0)"
echo "  imat_v5=$(du -h "${IMAT[v5]}"|cut -f1)  imat_both=$(du -h "${IMAT[both]}"|cut -f1)"

declare -A SM S
port=8360
build_and_smoke() { # $1 imat-label
  local imat="${IMAT[$1]}" out="$QST/${1}_CD-Q4_K_M.gguf" lg="$QST/.q_${1}.log"
  echo "==== build $1 (imat=$(basename "$imat")) $(date -u) ===="
  if [ ! -f "$out" ]; then
    "$BIN/llama-quantize" --imatrix "$imat" --tensor-type-file "$CD_MAP" "$F16" "$out" Q4_K_M >"$lg" 2>&1
    [ -f "$out" ] || { echo "  [FATAL] build $1"; tail -4 "$lg"; exit 1; }
  fi
  echo "  built $(du -h "$out"|cut -f1) $(grep -oE "[0-9.]+ BPW" "$lg" 2>/dev/null|tail -1)"
  # sanity smoke (k-quant floor -> expect 5/5 STOP; informational, not a gate)
  local res; res=$(bash "$SMOKE" "$out" "$port" "$GPU" 2048 2>&1)
  SM[$1]=$(echo "$res" | grep -oE "RESULT: [0-9]+/5 STOP" | head -1)
  echo "  smoke $1: ${SM[$1]:-?}"; port=$((port+1))
}

evalone() { # $1 imat-label $2 bench $3 port
  local served="$1_CD" gguf="$QST/$1_CD-Q4_K_M.gguf" tplf="$TPL/$2.yaml"
  echo "==== eval $served / $2 (port $3) $(date -u) ===="
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$OMK" --model "$gguf" --tokenizer "$TOK" \
    --template "$tplf" --backend llama --served-name "$served" \
    --results-dir "$RES" --port "$3" >"$QST/.eval_${served}_$2.log" 2>&1
  local sj; sj=$(find "$RES" -path "*$2*${served}*summary.json" 2>/dev/null | head -1)
  local sc="ERR"; [ -n "$sj" ] && sc=$("$PY" -c "import json;print(json.load(open('$sj')).get('score'))" 2>/dev/null)
  S[$1/$2]="$sc"; echo "  score=$sc  (summary=$sj)"
}

for c in v5 both; do build_and_smoke "$c"; done
eport=8370
for c in v5 both; do
  for bench in humanevalplus_full multipl_e_100; do
    evalone "$c" "$bench" "$eport"; eport=$((eport+1))
  done
done

echo; echo "######## CD-Q4_K_M (k-quant floor) IMATRIX QUALITY — v5 vs both ########"
printf "  %-8s | %-14s | %-10s | %-10s\n" imatrix smoke HE+ MPE-100
for c in v5 both; do
  printf "  %-8s | %-14s | %-10s | %-10s\n" "$c" "${SM[$c]:-?}" "${S[$c/humanevalplus_full]:-?}" "${S[$c/multipl_e_100]:-?}"
done
echo "  READ: both >= v5 within noise on BOTH benches -> adopt the larger imatrix as default"
echo "        (quality-neutral-or-better, free coverage win). both < v5 -> keep v5 imatrix."
rm -f "$QST"/*.gguf
echo "[done] $(date -u)"
