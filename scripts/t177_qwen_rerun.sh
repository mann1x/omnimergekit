#!/usr/bin/env bash
# T177.1 — re-run the benches that failed for Qwen3.6 due to the colliding
# "Question:"/"Problem:" until-stop, using the Qwen-tuned templates
# (gsm8k_100_qwen / math500_100_qwen). Method-identical to canonical; only the
# stop differs. Separate port (8094) so it can run alongside the main suite.
set -uo pipefail
LIMIT="${1:-0}"; ONLY="${2:-gsm8k_100_qwen,math500_100_qwen}"
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
PORT=8094
RESULTS=/srv/ml/eval_results_t177/qwen36mtp
LOGS=/srv/ml/logs/t177
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULTS" "$LOGS"
IFS="," read -ra TS_ARR <<<"$ONLY"
for t in "${TS_ARR[@]}"; do
  echo "[$(date -Iseconds)] ----- $t (limit=$LIMIT) -----"
  cmd=("$OMK_PY" "$OMK/eval/omk_eval.py" --backend llama --template "$t" \
       --quant iq3_xxs --model "$GGUF" --tokenizer "$TOK" \
       --served-name "$SERVED" --port "$PORT" --results-dir "$RESULTS")
  [[ "$LIMIT" -gt 0 ]] && cmd+=(--limit "$LIMIT")
  blog="$LOGS/t177_${t}_${TS}.log"
  echo "[$t] log=$blog"
  ( "${cmd[@]}" ) >"$blog" 2>&1; rc=$?
  summ="$RESULTS/$t/$SERVED/summary.json"
  sc="NO_RESULT"; [[ -f "$summ" ]] && sc=$($OMK_PY -c "import json;d=json.load(open(\"$summ\"));print(d.get(\"score\"),d.get(\"metric\"),d.get(\"filter\"))" 2>/dev/null)
  echo "[$(date -Iseconds)] [$t] rc=$rc score=$sc"
done
echo "[$(date -Iseconds)] QWEN_RERUN_DONE"
