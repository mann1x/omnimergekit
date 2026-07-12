#!/usr/bin/env bash
# T177 — Qwen3.6-35B-A3B-MTP UD-IQ3_XXS: 9-bench canonical + MPE100 on bs2 GPU1.
# bs2-native driver over omk_eval (the canonical unit). NOT a one-off recipe:
# identical per-bench omk_eval invocation as eval_suite_llama.sh, with bs2 paths,
# the Qwen3.6 tokenizer, and the MTP-capable llama.cpp build (af6528e, #23643).
# Plain decode (no --spec-type): the new binary makes the MTP GGUF LOAD cleanly,
# which is the requirement; spec-decode can't be wired per-bench via LLAMA_EXTRA
# (it would clobber the mandatory --reasoning-format flags). GPU1-pinned.
#
# Usage: t177_qwen36_suite.sh [--limit N] [--only csv] [--skip csv]
set -uo pipefail
LIMIT=0; ONLY=""; SKIP=""
while [[ $# -gt 0 ]]; do case "$1" in
  --limit) LIMIT="$2"; shift 2;;
  --only)  ONLY="$2";  shift 2;;
  --skip)  SKIP="$2";  shift 2;;
  *) echo "unknown arg: $1"; exit 2;;
esac; done

export CUDA_VISIBLE_DEVICES=1
export LLAMA_BIN=/srv/ml/repos/llama.cpp-latest/build/bin
OMK=/srv/ml/repos/omnimergekit
OMK_PY=/srv/ml/envs/envs/omnimergekit/bin/python
export OMK_PYTHON=$OMK_PY
export LM_EVAL_BIN=/srv/ml/envs/envs/omnimergekit/bin/lm-eval
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH
export HF_ALLOW_CODE_EVAL=1
[ -z "${HF_TOKEN:-}" ] && [ -f ~/.cache/huggingface/token ] && export HF_TOKEN=$(cat ~/.cache/huggingface/token)
GGUF=/mnt/sdc/ml/gguf/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
TOK=/mnt/sdc/ml/google/qwen36-35b-a3b-tok
SERVED=qwen36mtp_iq3xxs
PORT=8092
RESULTS=/srv/ml/eval_results_t177/qwen36mtp
LOGS=/srv/ml/logs/t177
TS=$(date +%Y%m%d_%H%M%S)
SUITE_LOG=$LOGS/t177_suite_${TS}.log
SUMMARY=$RESULTS/SUMMARY.md
mkdir -p "$RESULTS" "$LOGS"
log(){ echo "[$(date -Iseconds)] $*" | tee -a "$SUITE_LOG"; }

# 9 canonical + MPE100. LCB: lcb_medium_55 (Qwen is not a Gemma pruned variant;
# no _v4 parser recipe). multipl_e_100 is the MPE100 the user asked for.
TEMPLATES=(gpqa_diamond_full gsm8k_100 math500_100 aime_30 arc_challenge_full \
           ifeval_100 humaneval_full humanevalplus_full lcb_medium_55 multipl_e_100)
selected=()
if [[ -n "$ONLY" ]]; then IFS=',' read -ra a <<<"$ONLY"; selected=("${a[@]}"); else selected=("${TEMPLATES[@]}"); fi
if [[ -n "$SKIP" ]]; then IFS=',' read -ra sa <<<"$SKIP"; f=(); for k in "${selected[@]}"; do s=0; for x in "${sa[@]}"; do [[ "$k" == "$x" ]] && s=1; done; [[ $s -eq 0 ]] && f+=("$k"); done; selected=("${f[@]}"); fi

log "===== T177 Qwen3.6 MTP suite ====="
log "  gguf=$GGUF ($(stat -c %s "$GGUF"|numfmt --to=iec))  tok=$TOK"
log "  bin=$LLAMA_BIN  served=$SERVED  port=$PORT  gpu=1  limit=$LIMIT"
log "  templates: ${selected[*]}"
log "  results=$RESULTS"

declare -A SCORES DURS
S0=$(date +%s)
for t in "${selected[@]}"; do
  log "----- $t -----"
  out="$RESULTS/$t"; mkdir -p "$out"
  cmd=("$OMK_PY" "$OMK/eval/omk_eval.py" --backend llama --template "$t" \
       --quant iq3_xxs --model "$GGUF" --tokenizer "$TOK" \
       --served-name "$SERVED" --port "$PORT" --results-dir "$RESULTS")
  [[ "$LIMIT" -gt 0 ]] && cmd+=(--limit "$LIMIT")
  blog="$LOGS/t177_${t}_${TS}.log"
  b0=$(date +%s)
  log "[$t] $(printf '%q ' "${cmd[@]}")"
  ( "${cmd[@]}" ) >"$blog" 2>&1; rc=$?
  b1=$(date +%s); bd=$((b1-b0))
  summ="$out/$SERVED/summary.json"
  score="NO_RESULT"
  if [[ -f "$summ" ]]; then
    score=$(python3 -c "
import json,sys
d=json.load(open('$summ')); s=d.get('score')
if s is None: print('NO_SCORE'); sys.exit()
m=d.get('metric'); f=d.get('filter'); tag=(f'{m},{f}' if m and f else (m or 'score'))
print(f'{s*100:.2f}%  ({tag})' if isinstance(s,(int,float)) and s<=1.0 else f'{s}  ({tag})')" 2>/dev/null || echo PARSE_ERR)
  fi
  SCORES["$t"]="$score"; DURS["$t"]="$bd"
  log "[$t] rc=$rc dur=${bd}s score: $score  (log $blog)"
  pkill -KILL -f "port $PORT" 2>/dev/null || true; sleep 2
done
S1=$(date +%s)
{ echo "# T177 Qwen3.6-35B-A3B-MTP UD-IQ3_XXS — $TS"; echo;
  echo "GGUF: \`$GGUF\`  | bin: af6528e (#23643 MTP)  | backend: llama.cpp plain decode | gpu1";
  echo "Duration: $((S1-S0))s"; echo;
  echo "| Bench | Score | Dur(s) |"; echo "|---|---|---|";
  for t in "${selected[@]}"; do echo "| $t | ${SCORES[$t]:-MISSING} | ${DURS[$t]:-?} |"; done; } | tee "$SUMMARY"
log "SUMMARY -> $SUMMARY"
