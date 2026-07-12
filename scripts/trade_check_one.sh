#!/usr/bin/env bash
# trade_check_one.sh — T204: capability trade-check for the DERN-Eq.11 candidate.
# Runs HE+ / MPE-100 / IFEval-100 on ONE Q4_K_M GGUF, pinned to one GPU, via the
# canonical omk_eval engine (NOT lm_eval directly). Identical settings on both
# models (dern11 vs noswap-mytc) so the score delta is purely DERN's fold.
#
#   --parallel 2 : documented floor for llama backend when reasoning_budget>4096
#                  (avoids IFEval SAT_COLLAPSE). Same on both models => valid delta.
#
# Usage: trade_check_one.sh <variant> <gguf> <gpu_id> <port>
set -uo pipefail
VAR="${1:?variant}"; GGUF="${2:?gguf}"; GPU="${3:?gpu}"; PORT="${4:?port}"
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit
TOK=/srv/ml/models/base/gemma-4-26B-A4B-it          # 128e tokenizer (NOT the pruned dir)
RES=/srv/ml/eval_results_dern11_tradecheck
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
export LM_EVAL_BIN=/root/anaconda3/envs/omnimergekit/bin/lm-eval
export HF_ALLOW_CODE_EVAL=1                          # HE+/MPE code exec gate
ts(){ date '+%T %Z'; }
cd "$OMK"
echo "===== [$VAR] trade-check start $(ts)  gpu=$GPU port=$PORT gguf=$GGUF ====="
for TPL in humanevalplus_full multipl_e_100 ifeval_100; do
  echo "=== [$VAR] $TPL START $(ts) ==="
  "$PY" eval/omk_eval.py --model "$GGUF" --template "$TPL" --backend llama \
      --quant q4_k_m --gpus "$GPU" --parallel 2 --port "$PORT" \
      --results-dir "$RES" --served-name "$VAR" --tokenizer "$TOK" \
    || echo "=== [$VAR] $TPL FAILED rc=$? $(ts) ==="
  echo "=== [$VAR] $TPL END $(ts) ==="
done
echo "===== [$VAR] TRADECHECK DONE $(ts) ====="
