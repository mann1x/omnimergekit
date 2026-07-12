#!/usr/bin/env bash
# rerun_std16_heplus.sh — re-run ONLY the HumanEval+ half of the STD16 deploy-sampler quant
# eval. The orchestrator (gate_std16_he_mpe.sh) called omk_eval with --model <gguf> and NO
# --tokenizer, so lm_eval tried AutoTokenizer.from_pretrained(<the .gguf>) and crashed at
# construction (OSError: not a valid JSON file) -> rc=1, 0 samples, score=null on all 13 tiers.
# MPE-100 used the native backend (no HF tokenizer) and is VALID — untouched here.
# Fix: pass --tokenizer <STD16 bf16 dir> (a real HF tokenizer dir). 2-GPU claim pool, resumable.
set -uo pipefail
BM=/srv/ml
PY="$BM/envs/envs/omnimergekit/bin/python"
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
OMK="$BM/repos/omnimergekit/eval/omk_eval.py"
GG=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF
STEM=gemma-4-A4B-98e-v7-coder-it
TOK=/mnt/sdc/ml/t223_fk/STD16-bf16          # the fix: a valid HF tokenizer+config dir
WORK=/mnt/sdc/ml/std16_gate/he_mpe
mkdir -p "$WORK/logs" "$WORK/locks_he" "$WORK/done"
LOG="$WORK/rerun_heplus.log"
exec >>"$LOG" 2>&1
ts(){ date -u +%T; }
magic(){ head -c4 "$1" 2>/dev/null; }

# tier -> deploy-sampler temp (09|08), from the completed 48-seed loop-gate classification
declare -A TP=(
  [Q8_0]=08 [Q6_K_L]=09 [Q6_K]=09 [Q5_K_L]=09 [Q5_K_M]=09
  [Q4_K_L]=08 [Q4_K_M]=09 [Q4_K_S]=09 [IQ4_NL]=09 [IQ4_XS]=09
  [Q3_K_L]=08 [Q3_K_M]=09 [CD-Q2_K]=09 )
TIERS=(Q8_0 Q6_K_L Q6_K Q5_K_L Q5_K_M Q4_K_L Q4_K_M Q4_K_S IQ4_NL IQ4_XS Q3_K_L Q3_K_M CD-Q2_K)
# optional smoke mode: pass tier names as args to run only those
[ "$#" -gt 0 ] && TIERS=("$@")

[ -d "$TOK" ] && [ -f "$TOK/tokenizer.json" ] || { echo "FATAL tokenizer dir invalid: $TOK"; exit 1; }
echo "==================== STD16 HE+ RE-RUN (tokenizer fix) start $(ts) UTC  tiers=${TIERS[*]} ===================="

# clear the FAILED HE+ markers + crashed result dirs for the selected tiers (MPE untouched)
for T in "${TIERS[@]}"; do
  B="humanevalplus_full_minprep${TP[$T]}"
  rm -f "$WORK/done/${T}__${B}.done"
  rm -rf "$WORK/results/$T/$B"
done

run_he(){ # tier gpu port
  local T="$1"
  local G="$2"
  local P="$3"
  local B="humanevalplus_full_minprep${TP[$T]}"
  local gguf="$GG/$STEM-$T.gguf"
  local TD="$WORK/results/$T/$B"
  [ -f "$WORK/done/${T}__${B}.done" ] && { echo "[$(ts)] $T $B already done"; return 0; }
  [ "$(magic "$gguf")" = GGUF ] || { echo "[$(ts)] SKIP $T (bad gguf magic)"; return 0; }
  echo "[$(ts)] [GPU$G:$P] $T $B start (tok=$TOK)"
  CUDA_VISIBLE_DEVICES=$G "$PY" "$OMK" --model "$gguf" --tokenizer "$TOK" --template "$B" \
    --backend llama --quant gguf --port "$P" --results-dir "$TD" --served-name "STD16-$T" \
    > "$WORK/logs/${T}__${B}.rerun.log" 2>&1 || echo "[$(ts)] $T $B rc=$?"
  touch "$WORK/done/${T}__${B}.done"
  echo "[$(ts)] [GPU$G:$P] $T $B done  score=$(grep -oE 'score=[0-9.]+|score=None' "$WORK/logs/${T}__${B}.rerun.log" | tail -1)"
}
worker(){ # gpu port
  local G="$1"
  local P="$2"
  local T
  for T in "${TIERS[@]}"; do
    mkdir "$WORK/locks_he/$T.lock" 2>/dev/null || continue
    run_he "$T" "$G" "$P"
  done
}
rm -rf "$WORK/locks_he"/*.lock 2>/dev/null
NW="${NWORKERS:-4}"          # 4 workers = 2 per GPU (validated 99%-util Blackwell config)
for i in $(seq 0 $((NW - 1))); do
  worker "$((i % 2))" "$((8270 + i))" &
done
wait
echo "[$(ts)] ==================== STD16 HE+ RE-RUN DONE  $(ts) ===================="
echo "STD16_HEPLUS_RERUN_DONE"
