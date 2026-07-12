#!/usr/bin/env bash
# validate_imatrix_corpus_quality.sh — does the NEW (balanced-augmented) imatrix corpus
# improve K-quant QUALITY over v5 (and over no-imatrix) across the shipping tiers?
# This is the gate for "re-quantize ALL tiers with the new-corpus imatrix" (user 2026-06-06).
#
# "new imatrix" = `both` = calibration_datav5 ++ balanced_imatrix_v7 (573k tok; a SUPERSET of
# v5, so both-vs-v5 fairly tests "augment our corpus with balanced data". Balanced-alone is
# code-light -> not a shipping choice for a coder model, so it is not tested here.)
# Reuses prebuilt imat_v5.dat / imat_both.dat (imatrix compute was valid). Greedy, llama, GPU0.
#
# Tiers x imatrix variants (high tier won't discriminate — imatrix barely moves 6-bit; the
# 2-bit Q2_K is the discriminating cell, max imatrix sensitivity, low baseline w/ headroom):
#   Q6_K      : {v5, both}          high tier  — new vs v5 (likely inconclusive: HE+~0.927)
#   Q2_K      : {v5, both}          LOW tier   — the discriminator (Q2_K_L sweep=0.8659/0.8233)
#   Q4_K_M    : {noimat, v5, both}  mid tier   — does imatrix recover Q4_K_M? new vs v5 vs none
#   CD-Q4_K_M : {v5, both}          k-quant-floor map (cd_maps_v7_fixed/coder, 0 i-quant slots)
# Per variant: build -> sanity smoke -> HE+ (humanevalplus_full) -> MPE-100 (multipl_e_100) -> rm.
# All k-quants terminate (5/5 STOP), so MPE runs for every variant for a complete table; a
# variant whose HE+ collapses (<0.50) is flagged and MPE is skipped (genuine-break gate only).
set -uo pipefail
# omk_eval shells out to the `lm-eval` console script; a detached launch needs the env bin on PATH.
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH
GPU=${GPU:-0}
BIN=/opt/llama.cpp/build/bin
PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
TPL=/srv/ml/repos/omnimergekit/eval/templates
SCR=/srv/ml/repos/omnimergekit/scripts
SMOKE=/srv/ml/scripts/smoke_gguf.sh
TOK=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it
GD=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF
F16="$GD/gemma-4-A4B-98e-v7-coder-it-F16.gguf"
CD_MAP="$SCR/cd_maps_v7_fixed/coder/tensor_types_CD-Q4_K_M.txt"
IMAT_DIR=/mnt/sdc/ml/cd_fixed_v7/imat_matrix
declare -A IMAT; IMAT[v5]="$IMAT_DIR/imat_v5.dat"; IMAT[both]="$IMAT_DIR/imat_both.dat"
QST=${QST:-/mnt/sdc/ml/cd_fixed_v7/imat_quality}; mkdir -p "$QST"
RES=${RES:-/srv/ml/eval_results_imat_quality}
LOG=${LOG:-/srv/ml/logs/validate_imatrix_corpus_quality.txt}
: > "$LOG"; exec >>"$LOG" 2>&1

for p in "$F16" "$CD_MAP" "$OMK" "$TOK" "$SMOKE" "${IMAT[v5]}" "${IMAT[both]}" \
         "$TPL/humanevalplus_full.yaml" "$TPL/multipl_e_100.yaml"; do
  [ -e "$p" ] || { echo "[FATAL] missing $p"; exit 1; }
done
command -v lm-eval >/dev/null || { echo "[FATAL] lm-eval not on PATH"; exit 1; }
[ "$(grep -cE '=IQ' "$CD_MAP")" -eq 0 ] || { echo "[FATAL] CD_MAP has i-quant slots"; exit 1; }
echo "######## imatrix-corpus QUALITY across tiers $(date -u) ########"
echo "  new='both'(v5+balanced 573k) vs v5(421k) vs noimat ; HE+/MPE-100 greedy GPU$GPU ; lm-eval=$(command -v lm-eval)"

declare -A S BPW
sport=${SPORT:-8360}; eport=${EPORT:-8370}

build_one() { # $1 tier $2 variant -> $QST/${tier}__${variant}.gguf
  local tier="$1" v="$2" out="$QST/${tier}__${v}.gguf" lg="$QST/.q_${tier}__${v}.log"
  echo "==== build $tier / $v $(date -u) ===="
  if [ "$tier" = "CD-Q4_K_M" ]; then
    "$BIN/llama-quantize" --imatrix "${IMAT[$v]}" --tensor-type-file "$CD_MAP" "$F16" "$out" Q4_K_M >"$lg" 2>&1
  elif [ "$v" = "noimat" ]; then
    "$BIN/llama-quantize" "$F16" "$out" "$tier" >"$lg" 2>&1
  else
    "$BIN/llama-quantize" --imatrix "${IMAT[$v]}" "$F16" "$out" "$tier" >"$lg" 2>&1
  fi
  [ -f "$out" ] || { echo "  [FATAL] build $tier/$v"; tail -4 "$lg"; exit 1; }
  BPW[$tier/$v]=$(grep -oE "[0-9.]+ BPW" "$lg"|tail -1)
  echo "  built $(du -h "$out"|cut -f1) ${BPW[$tier/$v]}"
  local res; res=$(bash "$SMOKE" "$out" "$sport" "$GPU" 2048 2>&1)
  echo "  smoke $tier/$v: $(echo "$res"|grep -oE 'RESULT: [0-9]+/5 STOP'|head -1)"; sport=$((sport+1))
}

eval_one() { # $1 tier $2 variant $3 bench
  local tier="$1" v="$2" bench="$3" served="${tier}_${v}" gguf="$QST/${tier}__${v}.gguf"
  echo "==== eval $served / $bench (port $eport) $(date -u) ===="
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$OMK" --model "$gguf" --tokenizer "$TOK" \
    --template "$TPL/$bench.yaml" --backend llama --served-name "$served" \
    --results-dir "$RES" --port "$eport" >"$QST/.eval_${served}_${bench}.log" 2>&1
  # canonical path is $RES/$bench/$served/summary.json; the glob fallback requires a
  # trailing /summary.json so it never matches MultiPL-E per-lang results/*/_summary.json.
  local sj="$RES/$bench/$served/summary.json"
  [ -f "$sj" ] || sj=$(find "$RES" -path "*$bench*${served}*/summary.json" 2>/dev/null | head -1)
  local sc="ERR"; [ -n "$sj" ] && sc=$("$PY" -c "import json;print(json.load(open('$sj')).get('score'))" 2>/dev/null)
  S[$tier/$v/$bench]="$sc"; echo "  RESULT $served $bench score=$sc"
  eport=$((eport+1))
}

do_variant() { # $1 tier $2 variant : build -> HE+ -> (MPE unless HE+ collapsed) -> rm
  local tier="$1" v="$2"
  # RESUMABLE: if both bench summaries already exist on disk, skip build+eval and harvest.
  local hej="$RES/humanevalplus_full/${tier}_${v}/summary.json"
  local mpj="$RES/multipl_e_100/${tier}_${v}/summary.json"
  if [ -f "$hej" ] && [ -f "$mpj" ]; then
    S[$tier/$v/humanevalplus_full]=$("$PY" -c "import json;print(json.load(open('$hej')).get('score'))" 2>/dev/null)
    S[$tier/$v/multipl_e_100]=$("$PY" -c "import json;print(json.load(open('$mpj')).get('score'))" 2>/dev/null)
    echo "==== skip $tier / $v — cached HE+=${S[$tier/$v/humanevalplus_full]} MPE=${S[$tier/$v/multipl_e_100]} $(date -u) ===="
    return
  fi
  build_one "$tier" "$v"
  eval_one "$tier" "$v" humanevalplus_full
  local he="${S[$tier/$v/humanevalplus_full]}"
  if awk "BEGIN{exit !(\"$he\"+0 >= 0.50)}" 2>/dev/null; then
    eval_one "$tier" "$v" multipl_e_100
  else
    echo "  [skip MPE] $tier/$v HE+=$he collapsed (<0.50) — genuine break"; S[$tier/$v/multipl_e_100]="(he-collapse)"
  fi
  rm -f "$QST/${tier}__${v}.gguf"
}

# TIERS overridable so two GPU workers can split disjoint tier sets (no cell overlap = no race).
# Per tier: noimat vs v5 vs both, except CD-* which is imatrix-only (v5 vs both).
TIERS=${TIERS:-"Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K CD-Q4_K_M"}
echo "######## worker tiers: $TIERS  (GPU$GPU, ports s=$sport e=$eport, QST=$QST) ########"
for tier in $TIERS; do
  case "$tier" in CD-*) vars="v5 both";; *) vars="noimat v5 both";; esac
  echo; echo "########## TIER $tier — $vars (GPU$GPU) ##########"
  for v in $vars; do do_variant "$tier" "$v"; done
done

echo; echo "######## IMATRIX-CORPUS QUALITY SUMMARY (HE+ / MPE-100) — greedy ########"
printf "  %-11s | %-7s | %-9s | %-9s | %s\n" tier variant HE+ MPE-100 BPW
for cell in "Q6_K noimat" "Q6_K v5" "Q6_K both" "Q5_K_M noimat" "Q5_K_M v5" "Q5_K_M both" "Q4_K_M noimat" "Q4_K_M v5" "Q4_K_M both" "Q3_K_M noimat" "Q3_K_M v5" "Q3_K_M both" "Q2_K noimat" "Q2_K v5" "Q2_K both" "CD-Q4_K_M v5" "CD-Q4_K_M both"; do
  t="${cell% *}"; v="${cell#* }"
  # authoritative: read each score straight from disk (immune to the None-harvest + skip cases)
  hej="$RES/humanevalplus_full/${t}_${v}/summary.json"; mpj="$RES/multipl_e_100/${t}_${v}/summary.json"
  he=$([ -f "$hej" ] && "$PY" -c "import json;print(round(json.load(open('$hej')).get('score'),4))" 2>/dev/null || echo "?")
  mp=$([ -f "$mpj" ] && "$PY" -c "import json;print(round(json.load(open('$mpj')).get('score'),4))" 2>/dev/null || echo "?")
  printf "  %-11s | %-7s | %-9s | %-9s | %s\n" "$t" "$v" "${he:-?}" "${mp:-?}" "${BPW[$t/$v]:-?}"
done
echo "  READ: per tier, both>=v5 => new corpus helps. Q4_K_M: v5/both vs noimat => does imatrix"
echo "        recover Q4_K_M. If 'both' wins broadly -> re-quant ALL tiers with imat_both (user gate)."
echo "[done] $(date -u)"
