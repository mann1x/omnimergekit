#!/usr/bin/env bash
# Fresh reasoning-ON gsm8k_100 + math500_100 on v8 Q6 — canonical recipe (omk auto-sets
# --reasoning-format deepseek --reasoning-budget 12288), --parallel 8, exactly like the 128e
# b9700 suite run. Fresh dir => empty cache. Tests whether v8 completes under reasoning-ON
# without the resume x parallel edge case (i.e. whether v8 has a retry-unrecoverable peg-500
# problem that 128e lacked). GPU1, sequential. PID-kill only.
set -uo pipefail
export OMK_ROOT=/srv/ml/repos/omnimergekit-canonical
export LLAMA_BIN=/mnt/sdc/ml/llama.cpp-b9700/build/bin
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
GGUF=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-fkbroad-soft2-imat-Q6_K.gguf
TOK=/srv/ml/google/gemma-4-26B-A4B-it
RES=/srv/ml/v8_mathrerun_reasonon
run () {
  local gpu=$1 port=$2 tmpl=$3
  OMK_GPUS=$gpu OMK_GPU_WAIT_S=60 python "$OMK_ROOT/eval/omk_eval.py" \
    --backend llama --template "$tmpl" --quant q6_k \
    --model "$GGUF" --tokenizer "$TOK" --served-name v8rr_reasonon_q6k \
    --port "$port" --parallel 8 --results-dir "$RES" \
    > "/srv/ml/scripts/mathrerun_v8_reasonon_${tmpl}.run.log" 2>&1
  echo "[done $tmpl rc=$? $(date +%T)]"
}
run 1 8094 gsm8k_100
run 1 8094 math500_100
echo "[REASONON_MATH_DONE $(date +%T)]"
