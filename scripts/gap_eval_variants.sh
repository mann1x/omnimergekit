#!/usr/bin/env bash
# gap_eval_variants.sh — eval the science-restore variants on GPQA, score on G_gap,
# and print the full comparison table (anchor rungs + v8b-safe + pes13 + v8 final).
# Tests the dissection's prediction: restoring science (v8b-safe) recovers G_gap.
# v8b-safe on GPU0, pes13 (v8b-safe + PES) on GPU1, parallel. Greedy b9700, frozen template.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
export OMK_ROOT=/srv/ml/repos/omnimergekit-canonical
export OMK_TOKENIZER=/srv/ml/google/gemma-4-26B-A4B-it
export LLAMA_BIN=/mnt/sdc/ml/llama.cpp-b9700/build/bin
SUITE=$OMK_ROOT/eval/eval_suite_llama.sh
DIS=/mnt/sdc/ml/gpqa_dissect
GAP=$DIS/gpqa_gap.json
SCORER=$DIS/gpqa_score_subset.py
SFT=/mnt/sdc/ml/sft_heal
V8BSAFE=$SFT/gemma-4-A4B-98e-v7-coder-v8b-safe-soft2-imat-Q6_K.gguf
PES13=$SFT/gemma-4-A4B-98e-v7-coder-v8b-safe-pes13-soft2-imat-Q6_K.gguf
ts(){ date '+%T %Z'; }
echo "==================== G_gap eval of science-restore variants $(ts) ===================="
for f in "$V8BSAFE" "$PES13" "$GAP" "$SCORER" "$SUITE" "$LLAMA_BIN/llama-server"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done

run_gpqa(){ # gpu port variant gguf ws
  local gpu=$1 port=$2 variant=$3 gguf=$4 ws=$5
  echo "[$variant $(ts)] GPQA start GPU$gpu:$port"
  OMK_GPUS=$gpu OMK_GPU_WAIT_S=300 OMK_WS=$ws \
    bash "$SUITE" --variant "$variant" --gguf "$gguf" --port "$port" --only gpqa_diamond_full \
    > "$DIS/${variant}.run.log" 2>&1
  echo "[$variant $(ts)] GPQA done rc=$?"
}

( run_gpqa 0 8092 v8bsafe_q6 "$V8BSAFE" "$DIS/ws_v8bsafe" ) &
P1=$!
( run_gpqa 1 8093 v8bsafe_pes13_q6 "$PES13" "$DIS/ws_pes13" ) &
P2=$!
wait $P1 $P2
echo "[gapeval $(ts)] both done; scoring G_gap"

S="samples_gpqa_diamond_cot_zeroshot_*.jsonl"
R1="$DIS/ws_rung1/eval_results_llama_suite/rung1_v7coder_q6/gpqa_diamond_full/*/lm_eval_out/*/$S"
R2="$DIS/ws_rung2/eval_results_llama_suite/rung2_fkbroad_sel_q6/gpqa_diamond_full/*/lm_eval_out/*/$S"
SAFE="$DIS/ws_v8bsafe/eval_results_llama_suite/v8bsafe_q6/gpqa_diamond_full/*/lm_eval_out/*/$S"
PESS="$DIS/ws_pes13/eval_results_llama_suite/v8bsafe_pes13_q6/gpqa_diamond_full/*/lm_eval_out/*/$S"
V8="/srv/ml/v8_q6k_full/eval_results_llama_suite/v8_fkbroad_soft2_q6k/gpqa_diamond_full/*/lm_eval_out/*/$S"
echo "==================== G_gap RECOVERY TABLE $(ts) ===================="
"$PY" "$SCORER" "$GAP" \
  "rung1_v7coder_q6=$R1" \
  "rung2_fkbroad_sel(noDERN)=$R2" \
  "v8_final(sel+dern)=$V8" \
  "v8b_safe(+85sci)=$SAFE" \
  "v8b_safe_pes13=$PESS"
echo "==================== GAPEVAL DONE $(ts) ===================="
