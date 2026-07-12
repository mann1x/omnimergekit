#!/usr/bin/env bash
# build_eval_qat_lowbit.sh — v7 QAT low-bit / CD-qat investigation (items 2/3/4, 2026-06-07).
#
# Question: do low-bit quants off Google's QAT checkpoint beat vanilla-base low-bit
# (item 4, unsloth's 2-bit-from-QAT claim), and is CD-qat (CD contrib map + layer
# protection ON the QAT base) worth a tier (item 2, corrected)?
#
# Design: matched-corpus head-to-head. Both sides use the SAME 'both' imatrix corpus
# (calibration_datav5 ++ balanced_imatrix_v7), so the ONLY difference is the base
# (QAT-pruned vs vanilla). Vanilla-'both' baselines come from the crossover study
# (Q2_K 0.8476/0.79, Q3_K_M 0.9207/0.88, CD-Q4_K_M 0.9024/0.8767); the crossover
# never did IQ3_M, so we build one vanilla-'both' IQ3_M here for the headline rescue
# test (vanilla IQ3_M COLLAPSED in the sweep — does QAT rescue it?).
#
# Smoke-gated: a degenerate tier (<3/5 STOP) is recorded as collapse and the full
# eval is SKIPPED (avoids the 12h IQ3_M hang that bit the sweep). Greedy/canonical.
set -uo pipefail
GPU=${GPU:-0}
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
export PATH=$BM/envs/envs/omnimergekit/bin:${PATH:-}
BIN=/opt/llama.cpp/build/bin
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
TPL=$BM/repos/omnimergekit/eval/templates
SMOKE=$BM/scripts/smoke_gguf.sh
QATF16=/mnt/sdc/ml/qat_investig/v7coder-qat-F16.gguf
VANF16=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF/gemma-4-A4B-98e-v7-coder-it-F16.gguf
TOK=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it
CDMAP=$BM/scripts/cd_maps_v7_fixed/coder/tensor_types_CD-Q4_K_M.txt
IMAT_VAN=/mnt/sdc/ml/cd_fixed_v7/imat_matrix/imat_both.dat   # vanilla 'both' imatrix (prebuilt)
CALV5=$BM/repos/omnimergekit/scripts/calibration_datav5.txt
CALBAL=/mnt/sdc/ml/corpora/balanced_imatrix_v7.txt
WORK=/mnt/sdc/ml/qat_investig
CALIB=$WORK/calib_both.txt
IMAT_QAT=$WORK/imat_qat_both.dat
RES=/srv/ml/eval_results_qat_investig
LOG=$WORK/qat_lowbit_$(date -u +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
L(){ echo "[qatinv $(date -u +%H:%M:%S)] $*"; }
mkdir -p "$RES"

L "=== QAT low-bit / CD-qat investigation (GPU$GPU) ==="
for f in "$QATF16" "$VANF16" "$TOK" "$CDMAP" "$IMAT_VAN" "$CALV5" "$CALBAL" "$OMK" "$SMOKE" \
         "$BIN/llama-imatrix" "$BIN/llama-quantize" "$TPL/humanevalplus_full.yaml" "$TPL/multipl_e_100.yaml"; do
  [ -e "$f" ] || { L "FATAL missing $f"; exit 1; }
done
command -v lm-eval >/dev/null || { L "FATAL lm-eval not on PATH"; exit 1; }
[ "$(grep -cE '=IQ' "$CDMAP")" -eq 0 ] || { L "FATAL CD map has i-quant slots (want k-quant-floor)"; exit 1; }

# [0] 'both' corpus
if [ ! -s "$CALIB" ]; then cat "$CALV5" "$CALBAL" > "$CALIB"; L "[0] built 'both' corpus $(wc -l <"$CALIB") lines"; else L "[0] corpus exists"; fi

# [1] QAT imatrix (QAT F16, 'both' corpus)
if [ -s "$IMAT_QAT" ]; then L "[1] QAT imatrix exists, skip"; else
  L "[1] llama-imatrix on QAT F16 ..."
  CUDA_VISIBLE_DEVICES=$GPU "$BIN/llama-imatrix" -m "$QATF16" -f "$CALIB" -o "$IMAT_QAT" -ngl 99 2>&1 | tail -10
  [ -s "$IMAT_QAT" ] || { L "FATAL QAT imatrix build failed"; exit 1; }
  L "[1] QAT imatrix $(du -h "$IMAT_QAT"|cut -f1)"
fi

gguf_ok(){ local f="$1"; [ -s "$f" ] && [ "$("$PY" -c "print(open('$f','rb').read(4).decode('latin1'))" 2>/dev/null)" = GGUF ]; }
sport=8480; eport=8490
do_tier(){ # served base(qat|van) tier imat cdmap(-|path)
  local served="$1" base="$2" tier="$3" imat="$4" cd="$5"
  local out="$WORK/${served}.gguf" lg="$WORK/.q_${served}.log"
  local f16; [ "$base" = qat ] && f16="$QATF16" || f16="$VANF16"
  local hej="$RES/humanevalplus_full/${served}/summary.json"
  if [ -f "$hej" ]; then L "[$served] cached, skip"; return 0; fi
  if [ -f "$RES/.${served}.collapse" ]; then L "[$served] prior collapse, skip"; return 0; fi
  L "[$served] build (base=$base tier=$tier cd=$([ "$cd" = - ] && echo none || basename "$cd"))"
  if gguf_ok "$out"; then L "  gguf exists"; else
    if [ "$cd" != "-" ]; then "$BIN/llama-quantize" --imatrix "$imat" --tensor-type-file "$cd" "$f16" "$out" "$tier" >"$lg" 2>&1
    else "$BIN/llama-quantize" --imatrix "$imat" "$f16" "$out" "$tier" >"$lg" 2>&1; fi
    gguf_ok "$out" || { L "  FATAL build $served"; tail -4 "$lg"; return 1; }
  fi
  L "  built $(du -h "$out"|cut -f1) $(grep -oE '[0-9.]+ BPW' "$lg"|tail -1)"
  # smoke gate
  local res stop; res=$(bash "$SMOKE" "$out" "$sport" "$GPU" 2048 2>&1); sport=$((sport+1))
  stop=$(echo "$res"|grep -oE '[0-9]+/5 STOP'|head -1); L "  smoke: ${stop:-?}"
  if echo "${stop:-0/5}"|grep -qE '^[0-2]/5'; then
    L "  COLLAPSE $served (smoke ${stop}) — record + skip full eval"; echo "$stop" > "$RES/.${served}.collapse"; rm -f "$out"; return 0
  fi
  local b sj
  for b in humanevalplus_full multipl_e_100; do
    sj="$RES/$b/${served}/summary.json"
    if [ -f "$sj" ]; then L "  $b cached"; else
      L "  eval $served/$b (port $eport)"
      CUDA_VISIBLE_DEVICES=$GPU "$PY" "$OMK" --model "$out" --tokenizer "$TOK" --template "$TPL/$b.yaml" \
        --backend llama --served-name "$served" --results-dir "$RES" --port "$eport" >"$WORK/.eval_${served}_${b}.log" 2>&1
      eport=$((eport+1))
    fi
    [ -f "$sj" ] && L "  RESULT $served/$b = $($PY -c "import json;print(json.load(open('$sj')).get('score'))" 2>/dev/null)"
  done
  rm -f "$out"
}

# served | base | tier | imat | cdmap
ROWS=(
  "qatboth-IQ3_M|qat|IQ3_M|$IMAT_QAT|-"
  "vanboth-IQ3_M|van|IQ3_M|$IMAT_VAN|-"
  "qatboth-Q2_K|qat|Q2_K|$IMAT_QAT|-"
  "qatboth-Q3_K_M|qat|Q3_K_M|$IMAT_QAT|-"
  "qatboth-CD-Q4_K_M|qat|Q4_K_M|$IMAT_QAT|$CDMAP"
  "qatboth-CD-Q2_K|qat|Q2_K|$IMAT_QAT|/srv/ml/scripts/cd_maps_v7_fixed/coder/tensor_types_CD-Q2_K.txt"
)
for row in "${ROWS[@]}"; do IFS='|' read -r s b t i c <<<"$row"; do_tier "$s" "$b" "$t" "$i" "$c" || L "[$s] FAILED"; done

L "###### QAT_LOWBIT_INVESTIG_DONE ######"
L "=== RESULTS: QAT-base vs vanilla-'both' (HE+ / MPE-100, matched corpus) ==="
sc(){ local sv="$1" bn="$2"; local sj="$RES/$bn/${sv}/summary.json"; [ -f "$sj" ] && "$PY" -c "import json;print(round(json.load(open('$sj')).get('score'),4))" 2>/dev/null || ([ -f "$RES/.${sv}.collapse" ] && echo "COLLAPSE($(cat "$RES/.${sv}.collapse"))" || echo "-"); }
printf "  %-18s %-22s %-22s\n" "tier" "QAT  HE+/MPE" "vanilla-both HE+/MPE"
printf "  %-18s %-22s %-22s\n" "IQ3_M"     "$(sc qatboth-IQ3_M humanevalplus_full)/$(sc qatboth-IQ3_M multipl_e_100)" "$(sc vanboth-IQ3_M humanevalplus_full)/$(sc vanboth-IQ3_M multipl_e_100)"
printf "  %-18s %-22s %-22s\n" "Q2_K"      "$(sc qatboth-Q2_K humanevalplus_full)/$(sc qatboth-Q2_K multipl_e_100)" "0.8476/0.79 (crossover)"
printf "  %-18s %-22s %-22s\n" "Q3_K_M"    "$(sc qatboth-Q3_K_M humanevalplus_full)/$(sc qatboth-Q3_K_M multipl_e_100)" "0.9207/0.88 (crossover)"
printf "  %-18s %-22s %-22s\n" "CD-Q4_K_M" "$(sc qatboth-CD-Q4_K_M humanevalplus_full)/$(sc qatboth-CD-Q4_K_M multipl_e_100)" "0.9024/0.8767 (crossover)"
L "[done] $(date -u)"
