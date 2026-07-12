#!/usr/bin/env bash
# T179 — gemma-4-A4B-98e-v6-coder Q4_K_M: 10-bench canonical on bs2 GPU1.
# Gemma-4 engine d6be315 (/opt/llama.cpp). STANDARD gsm8k_100/math500_100
# (Gemma CoT never trips the "Question:"/"Problem:" stop; the qwen variants are
# Qwen-only). 128e tokenizer per CLAUDE.md. Same omk_eval unit as T177.
# Staged on solidpc backup_models/scripts/, rsync'd to bs2 /srv/ml/scripts/.
set -uo pipefail
LIMIT=0; ONLY=""; SKIP=""
while [[ $# -gt 0 ]]; do case "$1" in
  --limit) LIMIT="$2"; shift 2;;
  --only)  ONLY="$2";  shift 2;;
  --skip)  SKIP="$2";  shift 2;;
  *) echo "unknown arg: $1"; exit 2;;
esac; done

export CUDA_VISIBLE_DEVICES=1
export LLAMA_BIN=/opt/llama.cpp/build/bin
OMK=/srv/ml/repos/omnimergekit
OMK_PY=/srv/ml/envs/envs/omnimergekit/bin/python
export OMK_PYTHON=$OMK_PY
export LM_EVAL_BIN=/srv/ml/envs/envs/omnimergekit/bin/lm-eval
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH
export HF_ALLOW_CODE_EVAL=1
[ -z "${HF_TOKEN:-}" ] && [ -f ~/.cache/huggingface/token ] && export HF_TOKEN=$(cat ~/.cache/huggingface/token)

GGUF=/mnt/sdc/ml/gguf/v6coder/gemma-4-A4B-98e-v6-coder-it-Q4_K_M.gguf
TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
SERVED=v6coder_q4km
PORT=8095
RESULTS=/srv/ml/eval_results_t179/v6coder_q4km
LOGS=/srv/ml/logs/t179
TS=$(date +%Y%m%d_%H%M%S)
SUITE_LOG=$LOGS/t179_suite_${TS}.log
mkdir -p "$RESULTS" "$LOGS"
log(){ echo "[$(date -Iseconds)] $*" | tee -a "$SUITE_LOG"; }

TEMPLATES=(gpqa_diamond_full gsm8k_100 math500_100 aime_30 arc_challenge_full \
           ifeval_100 humaneval_full humanevalplus_full lcb_medium_55 multipl_e_100)
selected=()
if [[ -n "$ONLY" ]]; then IFS=',' read -ra a <<<"$ONLY"; selected=("${a[@]}"); else selected=("${TEMPLATES[@]}"); fi
if [[ -n "$SKIP" ]]; then IFS=',' read -ra sa <<<"$SKIP"; f=(); for k in "${selected[@]}"; do s=0; for x in "${sa[@]}"; do [[ "$k" == "$x" ]] && s=1; done; [[ $s -eq 0 ]] && f+=("$k"); done; selected=("${f[@]}"); fi

log "===== T179 v6-coder Q4_K_M suite ====="
log "  gguf=$GGUF"
log "  tok=$TOK bin=$LLAMA_BIN served=$SERVED port=$PORT gpu=1 limit=$LIMIT"
log "  templates: ${selected[*]}"
S0=$(date +%s)
for t in "${selected[@]}"; do
  log "----- $t -----"
  cmd=("$OMK_PY" "$OMK/eval/omk_eval.py" --backend llama --template "$t" \
       --quant q4_k_m --model "$GGUF" --tokenizer "$TOK" \
       --served-name "$SERVED" --port "$PORT" --results-dir "$RESULTS")
  [[ "$LIMIT" -gt 0 ]] && cmd+=(--limit "$LIMIT")
  blog="$LOGS/t179_${t}_${TS}.log"
  b0=$(date +%s)
  ( "${cmd[@]}" ) >"$blog" 2>&1; rc=$?
  b1=$(date +%s); bd=$((b1-b0))
  summ="$RESULTS/$t/$SERVED/summary.json"
  score="NO_RESULT"
  if [[ -f "$summ" ]]; then
    score=$("$OMK_PY" - "$summ" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1]))
s=d.get('score'); m=d.get('metric'); f=d.get('filter')
tag=(f"{m},{f}" if m and f else (m or 'score'))
print(f"{s*100:.2f}%  ({tag})" if isinstance(s,(int,float)) and s<=1.0 else f"{s}  ({tag})")
PYEOF
)
  fi
  log "[$t] rc=$rc dur=${bd}s score: $score  (log $blog)"
done
S1=$(date +%s)
log "SUITE DONE in $((S1-S0))s"
echo "T179_SUITE_DONE"
