#!/usr/bin/env bash
# dissect_gpqa_p1.sh — localize the v7-coder -> v8 GPQA -20pp to a recipe STEP.
# Scores each recipe rung on G_gap (63 questions v7-coder gets right, v8 wrong):
#   rung1 = v7-coder Q6           (published; SELECTION=v7 map, shared, no DERN)  -> 63/63 anchor + quant control
#   rung2 = fkbroad-sel Q6        (expert_drop(fkbroad)+shared, NO DERN)          -> isolates the SELECTION step
#   rung4 = v8 fkbroad-soft2 Q6   (sel + DERN soft2 + fx2-imat; samples exist)     -> 0/63 by construction
# rung2 split: high G_gap => DERN destroys chemistry (selection innocent); low => selection is the culprit.
# rung1 on GPU0 (no build) || rung2 build+eval on GPU1, in parallel. Greedy, b9700, frozen gpqa template.
# Self-scheduling, resumable, PID-kill only. plain-Q6 for rung2 (GPQA is quant-robust; selection effect >> +-2q).
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
LCPP=/mnt/sdc/ml/llama.cpp-latest
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
DROP=/srv/ml/scripts/v8coder_fkbroad_drop_map.json
DIS=/mnt/sdc/ml/gpqa_dissect
GAP=$DIS/gpqa_gap.json
SCORER=$DIS/gpqa_score_subset.py

export OMK_ROOT=/srv/ml/repos/omnimergekit-canonical
export OMK_TOKENIZER=/srv/ml/google/gemma-4-26B-A4B-it
export LLAMA_BIN=/mnt/sdc/ml/llama.cpp-b9700/build/bin
SUITE=$OMK_ROOT/eval/eval_suite_llama.sh

RUNG1_GGUF=/mnt/sdc/ml/sft_heal/pub_v7_gguf/gemma-4-A4B-98e-v7-coder-it-Q6_K.gguf
V8_SAMPLES="/srv/ml/v8_q6k_full/eval_results_llama_suite/v8_fkbroad_soft2_q6k/gpqa_diamond_full/*/lm_eval_out/*/samples_gpqa_diamond_cot_zeroshot_*.jsonl"
FKB=$DIS/gemma-4-A4B-98e-v7-coder-fkbroad-sel-it
F16=$DIS/rung2-fkbroad-sel-F16.gguf
RUNG2_GGUF=$DIS/gemma-4-A4B-98e-v7-coder-fkbroad-sel-Q6_K.gguf
ts(){ date '+%T %Z'; }
echo "==================== GPQA recipe dissection (P1) $(ts) ===================="

# preflight
for f in "$SRC/config.json" "$DROP" "$RUNG1_GGUF" "$GAP" "$SCORER" "$SCR/expert_drop.py" \
         "$RECIPE/router_shared_upweight.py" "$LCPP/convert_hf_to_gguf.py" \
         "$LCPP/build/bin/llama-quantize" "$SUITE" "$LLAMA_BIN/llama-server"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
echo "[preflight $(ts)] disk:"; df -h "$DIS" | tail -1

run_gpqa(){ # gpu port variant gguf ws
  local gpu=$1 port=$2 variant=$3 gguf=$4 ws=$5
  echo "[$variant $(ts)] GPQA start GPU$gpu:$port ws=$ws"
  OMK_GPUS=$gpu OMK_GPU_WAIT_S=300 OMK_WS=$ws \
    bash "$SUITE" --variant "$variant" --gguf "$gguf" --port "$port" --only gpqa_diamond_full \
    > "$DIS/${variant}.run.log" 2>&1
  echo "[$variant $(ts)] GPQA done rc=$?"
}

# --- rung-1: v7-coder Q6 (exists) on GPU0, background ---
( run_gpqa 0 8092 rung1_v7coder_q6 "$RUNG1_GGUF" "$DIS/ws_rung1" ) &
P1=$!

# --- rung-2: build fkbroad-sel (no DERN) then GPQA on GPU1 ---
if [ ! -f "$RUNG2_GGUF" ]; then
  if [ ! -f "$FKB/model.safetensors" ] && [ ! -f "$FKB/model.safetensors.index.json" ]; then
    echo "[rung2 $(ts)] expert_drop(fkbroad) -> $FKB"
    "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$DROP" --output-dir "$FKB" \
      || { echo "FATAL expert_drop"; kill $P1 2>/dev/null; exit 3; }
    [ -f "$FKB/tokenizer.json" ] || { echo "FATAL tokenizer.json missing"; exit 3; }
  fi
  if [ ! -f "$FKB/.shared_applied" ]; then
    echo "[rung2 $(ts)] router_shared_upweight a=1.2"
    "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$FKB" \
      --alpha 1.2 --target mlp.down_proj.weight || { echo "FATAL shared"; exit 4; }
    touch "$FKB/.shared_applied"
  fi
  [ -f "$F16" ] || { echo "[rung2 $(ts)] convert F16";
    "$PY" "$LCPP/convert_hf_to_gguf.py" "$FKB" --outfile "$F16" --outtype f16 \
      || { echo "FATAL convert"; exit 5; }; }
  echo "[rung2 $(ts)] drop FKB bf16 to free disk"; rm -rf "$FKB"
  echo "[rung2 $(ts)] quantize plain Q6_K"
  "$LCPP/build/bin/llama-quantize" "$F16" "$RUNG2_GGUF" Q6_K 32 || { echo "FATAL quant"; exit 6; }
  magic=$("$PY" -c "print(open('$RUNG2_GGUF','rb').read(4).decode('latin1'))" 2>/dev/null)
  [ "$magic" = "GGUF" ] || { echo "FATAL bad GGUF"; exit 6; }
  rm -f "$F16"
fi
( run_gpqa 1 8093 rung2_fkbroad_sel_q6 "$RUNG2_GGUF" "$DIS/ws_rung2" ) &
P2=$!

wait $P1 $P2
echo "[dissect $(ts)] both GPQA done; scoring G_gap"

R1="$DIS/ws_rung1/eval_results_llama_suite/rung1_v7coder_q6/gpqa_diamond_full/*/lm_eval_out/*/samples_gpqa_diamond_cot_zeroshot_*.jsonl"
R2="$DIS/ws_rung2/eval_results_llama_suite/rung2_fkbroad_sel_q6/gpqa_diamond_full/*/lm_eval_out/*/samples_gpqa_diamond_cot_zeroshot_*.jsonl"
echo "==================== DISSECTION RESULT $(ts) ===================="
"$PY" "$SCORER" "$GAP" \
  "rung1_v7coder_q6=$R1" \
  "rung2_fkbroad_sel(noDERN)=$R2" \
  "rung4_v8_final=$V8_SAMPLES"
echo "==================== DISSECTION DONE $(ts) ===================="
