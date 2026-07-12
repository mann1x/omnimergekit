#!/usr/bin/env bash
# std16_body_fill.sh — fill the 2 body benches std16_q6k_full lacks: greedy LCB-100 + MPE-100,
# same config (b9700, ctx32768/budget8192, GREEDY) on STD16 publish Q6_K, GPU0.
set -uo pipefail
BM=/srv/ml
PY="$BM/envs/envs/omnimergekit/bin/python"
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
OMK="$BM/repos/omnimergekit/eval/omk_eval.py"
GGUF=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF/gemma-4-A4B-98e-v7-coder-it-Q6_K.gguf
TOK=/mnt/sdc/ml/t223_fk/STD16-bf16
WORK=/mnt/sdc/ml/std16_gate/body_fill
mkdir -p "$WORK/results" "$WORK/logs"
exec >>"$WORK/run.log" 2>&1
ts(){ date -u +%T; }
echo "==================== STD16 body-fill (LCB-100 + MPE-100, greedy, GPU0) start $(ts) ===================="
for B in lcb_medium_100_v4 multipl_e_100; do
  echo "[$(ts)] $B start"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$OMK" --model "$GGUF" --tokenizer "$TOK" --template "$B" \
    --backend llama --quant gguf --port 8091 --results-dir "$WORK/results" --served-name STD16 \
    > "$WORK/logs/${B}.log" 2>&1
  echo "[$(ts)] $B end rc=$? score=$(grep -oE 'score=[0-9.]+|score=None' "$WORK/logs/${B}.log" | tail -1)"
done
echo "STD16_BODY_FILL_DONE $(ts)"
