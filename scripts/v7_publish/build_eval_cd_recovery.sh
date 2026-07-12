#!/usr/bin/env bash
# build_eval_cd_recovery.sh — LANE B (item 3, 2026-06-07): can protected-CD maps RECOVER
# the failing flat low-bit tiers on the VANILLA base?
#
# Flat low-bit collapses: IQ3_M (0/5 smoke), IQ2_S/M (collapse), Q2_K (noimat collapse /
# 0.8476 with imatrix). The fixed CD maps protect the 120 critical tensors (attn_v/k +
# ffn_down x30 layers — the #563 rumination culprit) at k-quant while compressing the
# redundant body. This tests whether that protection brings the collapsing tiers back,
# and at what size/quality. Vanilla base + 'both' imatrix; smoke-gated; greedy/canonical.
#
# Compare each to its FLAT counterpart:
#   CD-Q2_K     vs flat Q2_K   (van-both 0.8476/0.79 ; noimat collapse)
#   CD-Q3_K_L   vs flat Q3_K_L (0.9146/0.8833)
#   CD-IQ2_K    vs flat IQ2_S/M (collapse) / IQ2_XS 0.7378
#   CD-IQ3_K_M  vs flat IQ3_M  (COLLAPSE)   <-- headline
set -uo pipefail
GPU=${GPU:-1}
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
export PATH=$BM/envs/envs/omnimergekit/bin:${PATH:-}
BIN=/opt/llama.cpp/build/bin
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
TPL=$BM/repos/omnimergekit/eval/templates
SMOKE=$BM/scripts/smoke_gguf.sh
VANF16=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF/gemma-4-A4B-98e-v7-coder-it-F16.gguf
TOK=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it
IMAT=/mnt/sdc/ml/cd_fixed_v7/imat_matrix/imat_both.dat
MD=$BM/scripts/cd_maps_v7_fixed/coder
WORK=/mnt/sdc/ml/cd_recovery
RES=/srv/ml/eval_results_cd_recovery
mkdir -p "$WORK" "$RES"
LOG=$WORK/cd_recovery_$(date -u +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
L(){ echo "[cdrec $(date -u +%H:%M:%S)] $*"; }

L "=== Lane B: protected-CD recovery of failing low-bit (vanilla base, GPU$GPU) ==="
for f in "$VANF16" "$TOK" "$IMAT" "$OMK" "$SMOKE" "$BIN/llama-quantize" \
         "$TPL/humanevalplus_full.yaml" "$TPL/multipl_e_100.yaml"; do
  [ -e "$f" ] || { L "FATAL missing $f"; exit 1; }
done
command -v lm-eval >/dev/null || { L "FATAL lm-eval not on PATH"; exit 1; }

gguf_ok(){ local f="$1"; [ -s "$f" ] && [ "$("$PY" -c "print(open('$f','rb').read(4).decode('latin1'))" 2>/dev/null)" = GGUF ]; }
sport=8500; eport=8510
do_tier(){ # served basetier cdmap
  local served="$1" tier="$2" cd="$3"
  local out="$WORK/${served}.gguf" lg="$WORK/.q_${served}.log"
  local hej="$RES/humanevalplus_full/${served}/summary.json"
  [ -f "$hej" ] && { L "[$served] cached, skip"; return 0; }
  [ -f "$RES/.${served}.collapse" ] && { L "[$served] prior collapse, skip"; return 0; }
  [ -f "$cd" ] || { L "[$served] FATAL map missing $cd"; return 1; }
  L "[$served] build (tier=$tier map=$(basename "$cd") iq_slots=$(grep -cE '=IQ' "$cd"))"
  if ! gguf_ok "$out"; then
    "$BIN/llama-quantize" --imatrix "$IMAT" --tensor-type-file "$cd" "$VANF16" "$out" "$tier" >"$lg" 2>&1
    gguf_ok "$out" || { L "  FATAL build $served"; tail -4 "$lg"; return 1; }
  fi
  L "  built $(du -h "$out"|cut -f1) $(grep -oE '[0-9.]+ BPW' "$lg"|tail -1)"
  local res stop; res=$(bash "$SMOKE" "$out" "$sport" "$GPU" 2048 2>&1); sport=$((sport+1))
  stop=$(echo "$res"|grep -oE '[0-9]+/5 STOP'|head -1); L "  smoke: ${stop:-?}"
  if echo "${stop:-0/5}"|grep -qE '^[0-2]/5'; then
    L "  COLLAPSE $served (smoke ${stop}) — record + skip full eval"; echo "$stop" > "$RES/.${served}.collapse"; rm -f "$out"; return 0
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

# served | base tier | cd map
ROWS=(
  "vancd-IQ3_K_M|IQ3_M|$MD/tensor_types_CD-IQ3_K_M.txt"
  "vancd-IQ2_K|IQ2_M|$MD/tensor_types_CD-IQ2_K.txt"
  "vancd-Q2_K|Q2_K|$MD/tensor_types_CD-Q2_K.txt"
  "vancd-Q3_K_L|Q3_K_L|$MD/tensor_types_CD-Q3_K_L.txt"
)
for row in "${ROWS[@]}"; do IFS='|' read -r s t c <<<"$row"; do_tier "$s" "$t" "$c" || L "[$s] FAILED"; done

L "###### CD_RECOVERY_DONE ######"
sc(){ local sv="$1" bn="$2"; local sj="$RES/$bn/${sv}/summary.json"; [ -f "$sj" ] && "$PY" -c "import json;print(round(json.load(open('$sj')).get('score'),4))" 2>/dev/null || ([ -f "$RES/.${sv}.collapse" ] && echo "COLLAPSE($(cat "$RES/.${sv}.collapse"))" || echo "-"); }
L "=== LANE B RESULTS: protected-CD (vanilla) vs flat counterpart (HE+ / MPE-100) ==="
printf "  %-16s %-24s %-26s\n" "tier" "CD-protected HE+/MPE" "flat counterpart"
printf "  %-16s %-24s %-26s\n" "CD-IQ3_K_M" "$(sc vancd-IQ3_K_M humanevalplus_full)/$(sc vancd-IQ3_K_M multipl_e_100)" "flat IQ3_M = COLLAPSE(0/5)"
printf "  %-16s %-24s %-26s\n" "CD-IQ2_K"   "$(sc vancd-IQ2_K humanevalplus_full)/$(sc vancd-IQ2_K multipl_e_100)" "flat IQ2_S/M = collapse"
printf "  %-16s %-24s %-26s\n" "CD-Q2_K"    "$(sc vancd-Q2_K humanevalplus_full)/$(sc vancd-Q2_K multipl_e_100)" "flat Q2_K = 0.8476/0.79"
printf "  %-16s %-24s %-26s\n" "CD-Q3_K_L"  "$(sc vancd-Q3_K_L humanevalplus_full)/$(sc vancd-Q3_K_L multipl_e_100)" "flat Q3_K_L = 0.9146/0.8833"
L "[done] $(date -u)"
