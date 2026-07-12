#!/usr/bin/env bash
# build_eval_coderx_tiers.sh — coderx (fs2440) counterparts of the v7-coder new
# low-bit tiers, so BOTH models ship the same set (user 2026-06-07). Each variant
# is smoke-gated + HE+/MPE100-evaluated; a coderx tier that degenerates is dropped.
#
# Tiers (matched to the coder lane):
#   cx-CD-qat-Q4_K_M  QAT base  + CD-Q4_K_M map + QAT imatrix   (coder: 0.9268/0.8767)
#   cx-CD-Q2_K        vanilla   + CD-Q2_K   map + vanilla imat  (coder: 0.9146/0.8767)
#   cx-CD-Q3_K_L      vanilla   + CD-Q3_K_L map + vanilla imat  (coder: 0.9146/0.8967)
#   cx-CD-qat-Q2_K    QAT base  + CD-Q2_K   map + QAT imatrix   (coder: 0.9146/MPE pending)
#
# Self-gates on the coderx QAT imatrix (built separately on GPU1), then builds the
# coderx vanilla 'both' imatrix (coder's imat_both.dat is CODER weights — invalid here).
# Greedy/canonical. imatrix mandatory (CD bodies are sub-Q4 / 2-bit).
set -uo pipefail
GPU=${GPU:-1}
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
export PATH=$BM/envs/envs/omnimergekit/bin:${PATH:-}
BIN=/opt/llama.cpp/build/bin
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
TPL=$BM/repos/omnimergekit/eval/templates
SMOKE=$BM/scripts/smoke_gguf.sh
W=/mnt/sdc/ml/qat_investig
QATF16=$W/v7coderx-qat-F16.gguf
VANF16=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it-GGUF/gemma-4-A4B-98e-v7-coderx-it-F16.gguf
TOK=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it
MD=$BM/scripts/cd_maps_v7_fixed/coderx
CALIB=$W/calib_both.txt
IMAT_QAT=$W/imat_qat_coderx_both.dat
IMAT_VAN=$W/imat_van_coderx_both.dat
RES=/srv/ml/eval_results_coderx_tiers
WORK=/mnt/sdc/ml/coderx_tiers
mkdir -p "$RES" "$WORK"
LOG=$WORK/coderx_tiers_$(date -u +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
L(){ echo "[cxtier $(date -u +%H:%M:%S)] $*"; }

L "=== coderx new-tier build+eval (GPU$GPU) ==="
for f in "$QATF16" "$VANF16" "$TOK" "$CALIB" "$OMK" "$SMOKE" "$BIN/llama-quantize" "$BIN/llama-imatrix" \
         "$MD/tensor_types_CD-Q2_K.txt" "$MD/tensor_types_CD-Q3_K_L.txt" "$MD/tensor_types_CD-Q4_K_M.txt" \
         "$TPL/humanevalplus_full.yaml" "$TPL/multipl_e_100.yaml"; do
  [ -e "$f" ] || { L "FATAL missing $f"; exit 1; }
done
command -v lm-eval >/dev/null || { L "FATAL lm-eval not on PATH"; exit 1; }

# [gate] wait for the coderx QAT imatrix (built separately); don't proceed while its builder runs
L "[gate] waiting on coderx QAT imatrix $IMAT_QAT"
for i in $(seq 1 320); do
  if [ -s "$IMAT_QAT" ] && ! pgrep -f "llama-imatrix.*v7coderx-qat-F16" >/dev/null; then break; fi
  sleep 15
done
[ -s "$IMAT_QAT" ] || { L "FATAL coderx QAT imatrix missing after wait"; exit 1; }
L "[gate] QAT imatrix ready $(du -h "$IMAT_QAT"|cut -f1)"

# [imat] coderx vanilla 'both' imatrix (matched corpus, coderx weights)
if [ -s "$IMAT_VAN" ]; then L "[imat] vanilla 'both' imat exists"; else
  L "[imat] build coderx vanilla 'both' imatrix on GPU$GPU"
  CUDA_VISIBLE_DEVICES=$GPU "$BIN/llama-imatrix" -m "$VANF16" -f "$CALIB" -o "$IMAT_VAN" -ngl 99 >"$WORK/imat_van.log" 2>&1
  [ -s "$IMAT_VAN" ] || { L "FATAL coderx vanilla imatrix failed"; tail -5 "$WORK/imat_van.log"; exit 1; }
  L "[imat] vanilla imat $(du -h "$IMAT_VAN"|cut -f1)"
fi

gguf_ok(){ local f="$1"; [ -s "$f" ] && [ "$("$PY" -c "print(open('$f','rb').read(4).decode('latin1'))" 2>/dev/null)" = GGUF ]; }
sport=8520; eport=8530
do_tier(){ # served f16 tier imat cdmap
  local served="$1" f16="$2" tier="$3" imat="$4" cd="$5"
  local out="$WORK/${served}.gguf" lg="$WORK/.q_${served}.log"
  local hej="$RES/humanevalplus_full/${served}/summary.json"
  [ -f "$hej" ] && { L "[$served] cached, skip"; return 0; }
  [ -f "$RES/.${served}.collapse" ] && { L "[$served] prior collapse, skip"; return 0; }
  [ -f "$cd" ] || { L "[$served] FATAL map missing $cd"; return 1; }
  L "[$served] build (tier=$tier map=$(basename "$cd"))"
  if ! gguf_ok "$out"; then
    "$BIN/llama-quantize" --imatrix "$imat" --tensor-type-file "$cd" "$f16" "$out" "$tier" >"$lg" 2>&1
    gguf_ok "$out" || { L "  FATAL build $served"; tail -4 "$lg"; return 1; }
  fi
  L "  built $(du -h "$out"|cut -f1) $(grep -oE '[0-9.]+ BPW' "$lg"|tail -1)"
  local res stop; res=$(bash "$SMOKE" "$out" "$sport" "$GPU" 2048 2>&1); sport=$((sport+1))
  stop=$(echo "$res"|grep -oE '[0-9]+/5 STOP'|head -1); L "  smoke: ${stop:-?}"
  if echo "${stop:-0/5}"|grep -qE '^[0-2]/5'; then
    L "  COLLAPSE $served (smoke ${stop}) — record + skip"; echo "$stop" > "$RES/.${served}.collapse"; rm -f "$out"; return 0
  fi
  local b sj
  for b in humanevalplus_full multipl_e_100; do
    sj="$RES/$b/${served}/summary.json"
    if [ ! -f "$sj" ]; then
      L "  eval $served/$b (port $eport)"
      CUDA_VISIBLE_DEVICES=$GPU "$PY" "$OMK" --model "$out" --tokenizer "$TOK" --template "$TPL/$b.yaml" \
        --backend llama --served-name "$served" --results-dir "$RES" --port "$eport" >"$WORK/.eval_${served}_${b}.log" 2>&1
      eport=$((eport+1))
    fi
    [ -f "$sj" ] && L "  RESULT $served/$b = $($PY -c "import json;print(json.load(open('$sj')).get('score'))" 2>/dev/null)"
  done
  rm -f "$out"
}

# cx-CD-qat-Q2_K DROPPED: coder proved it strictly dominated by vanilla CD-Q2_K
# (HE+ 0.9146 tie, MPE 0.85 < 0.8767, same 8.3 GB). QAT doesn't help CD-protected 2-bit.
do_tier cx-CD-qat-Q4_K_M "$QATF16" Q4_K_M "$IMAT_QAT" "$MD/tensor_types_CD-Q4_K_M.txt" || L "[cx-CD-qat-Q4_K_M] FAILED"
do_tier cx-CD-Q2_K       "$VANF16" Q2_K   "$IMAT_VAN" "$MD/tensor_types_CD-Q2_K.txt"   || L "[cx-CD-Q2_K] FAILED"
do_tier cx-CD-Q3_K_L     "$VANF16" Q3_K_L "$IMAT_VAN" "$MD/tensor_types_CD-Q3_K_L.txt" || L "[cx-CD-Q3_K_L] FAILED"

L "###### CODERX_TIERS_DONE ######"
sc(){ local sv="$1" bn="$2"; local sj="$RES/$bn/${sv}/summary.json"; [ -f "$sj" ] && "$PY" -c "import json;print(round(json.load(open('$sj')).get('score'),4))" 2>/dev/null || ([ -f "$RES/.${sv}.collapse" ] && echo "COLLAPSE($(cat "$RES/.${sv}.collapse"))" || echo "-"); }
L "=== CODERX RESULTS (HE+ / MPE-100) vs coder ==="
printf "  %-18s %-22s %-22s\n" "tier" "coderx HE+/MPE" "coder HE+/MPE"
printf "  %-18s %-22s %-22s\n" "CD-qat-Q4_K_M" "$(sc cx-CD-qat-Q4_K_M humanevalplus_full)/$(sc cx-CD-qat-Q4_K_M multipl_e_100)" "0.9268/0.8767"
printf "  %-18s %-22s %-22s\n" "CD-Q2_K"       "$(sc cx-CD-Q2_K humanevalplus_full)/$(sc cx-CD-Q2_K multipl_e_100)" "0.9146/0.8767"
printf "  %-18s %-22s %-22s\n" "CD-Q3_K_L"     "$(sc cx-CD-Q3_K_L humanevalplus_full)/$(sc cx-CD-Q3_K_L multipl_e_100)" "0.9146/0.8967"
L "[done] $(date -u)"
