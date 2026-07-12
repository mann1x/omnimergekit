#!/usr/bin/env bash
# eval_v8_lsfold.sh — T222 A/B eval (GPU1, b9700): GPQA-198 + G_gap subset + HE+/MPE for the
# LS-fold imat-Q6. Polls for the Q6 the build (build_v8_lsfold.sh, GPU0) produces, then:
#   1. GPQA_diamond_full via eval_suite_llama.sh (greedy, frozen template) — same harness that
#      produced the rung ladder, so the lsfold rung is apples-to-apples.
#   2. score on gpqa_gap.json (63 doc_ids) ALONGSIDE the existing rungs — the decisive readout:
#      v8-average shuffles gap 18->0 (survivor perturbation); does LS-fold (gate_up preserved)
#      hold gap>0 and aggregate>=99/198?
#   3. HE+164 + MPE-100 via omk_eval — must hold >= v8-average (93.29 / 89.33).
# GPU1, b9700 eval binary. PID-kill only.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
export OMK_ROOT=/srv/ml/repos/omnimergekit-canonical
export OMK_TOKENIZER=/srv/ml/google/gemma-4-26B-A4B-it          # GPQA tok (matches rung ladder)
export LLAMA_BIN=/mnt/sdc/ml/llama.cpp-b9700/build/bin
SUITE=$OMK_ROOT/eval/eval_suite_llama.sh
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it                       # code tok (matches reg_q40 ctrl)
DIS=/mnt/sdc/ml/gpqa_dissect
GAP=$DIS/gpqa_gap.json
SCORER=$DIS/gpqa_score_subset.py
SFT=/mnt/sdc/ml/sft_heal
Q6=$SFT/gemma-4-A4B-98e-v7-coder-fkbroad-lsfold-imat-Q6_K.gguf
NAME=v8-lsfold-imatq6
WS=$DIS/ws_lsfold
TD=$SFT/lsfold_codecheck
GPU=1
ts(){ date '+%T %Z'; }
echo "==================== eval v8 LS-FOLD (GPU$GPU, b9700) $(ts) ===================="
for f in "$GAP" "$SCORER" "$SUITE" "$OMK" "$LLAMA_BIN/llama-server" "$OMK_TOKENIZER/config.json" "$SRC/config.json"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done

# ── wait for the build to produce the imat-Q6 (up to ~3h), require stable GGUF header ──
echo "[wait $(ts)] polling for $Q6"
for i in $(seq 1 360); do
  if [ -f "$Q6" ]; then
    magic=$("$PY" -c "import sys;print(open('$Q6','rb').read(4).decode('latin1'))" 2>/dev/null)
    sz1=$(stat -c%s "$Q6" 2>/dev/null); sleep 20; sz2=$(stat -c%s "$Q6" 2>/dev/null)
    [ "$magic" = "GGUF" ] && [ "$sz1" = "$sz2" ] && [ "${sz1:-0}" -gt 1000000000 ] && break
  fi
  [ $((i % 10)) -eq 0 ] && echo "[wait $(ts)] not ready ($i/360)"
  sleep 30
done
[ -f "$Q6" ] || { echo "FATAL Q6 never appeared"; exit 3; }
echo "[wait $(ts)] Q6 ready: $(du -h "$Q6"|cut -f1)"

# ── 1. GPQA_diamond_full (greedy, frozen template) ───────
echo "[1 $(ts)] GPQA_diamond_full lsfold GPU$GPU:8094 -> $WS"
OMK_GPUS=$GPU OMK_GPU_WAIT_S=600 OMK_WS=$WS \
  bash "$SUITE" --variant "${NAME}" --gguf "$Q6" --port 8094 --only gpqa_diamond_full \
  > "$DIS/lsfold_q6.run.log" 2>&1
echo "[1 $(ts)] GPQA done rc=$?"

# ── 2. G_gap recovery table (add the lsfold rung next to the ladder) ──
S="samples_gpqa_diamond_cot_zeroshot_*.jsonl"
R1="$DIS/ws_rung1/eval_results_llama_suite/rung1_v7coder_q6/gpqa_diamond_full/*/lm_eval_out/*/$S"
R2="$DIS/ws_rung2/eval_results_llama_suite/rung2_fkbroad_sel_q6/gpqa_diamond_full/*/lm_eval_out/*/$S"
V8="/srv/ml/v8_q6k_full/eval_results_llama_suite/v8_fkbroad_soft2_q6k/gpqa_diamond_full/*/lm_eval_out/*/$S"
LS="$WS/eval_results_llama_suite/${NAME}/gpqa_diamond_full/*/lm_eval_out/*/$S"
echo "==================== G_gap RECOVERY TABLE (with LS-fold) $(ts) ===================="
"$PY" "$SCORER" "$GAP" \
  "rung1_v7coder_q6=$R1" \
  "rung2_fkbroad_sel(noDERN)=$R2" \
  "v8_final(sel+dern_avg)=$V8" \
  "lsfold(sel+dern_LS)=$LS" || echo "[2] WARN scorer rc=$?"

# ── 3. HE+164 + MPE-100 (greedy, --parallel 2) ───────────
mkdir -p "$TD"
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
export HF_ALLOW_CODE_EVAL=1
p=8295
for TPL in humanevalplus_full multipl_e_100; do
  echo "[3 $(ts)] $TPL (GPU$GPU:$p)"
  LLAMA_BIN="$LLAMA_BIN" CUDA_VISIBLE_DEVICES=$GPU "$PY" "$OMK" \
    --model "$Q6" --template "$TPL" --backend llama --quant gguf \
    --port "$p" --results-dir "$TD" --served-name "$NAME" \
    --tokenizer "$SRC" --parallel 2 || echo "[3] WARN $TPL rc=$?"
  p=$((p+1))
done

# ── 4. summary ───────────────────────────────────────────
echo "[4 $(ts)] === LS-FOLD EVAL DONE ==="
for TPL in humanevalplus_full multipl_e_100; do
  Sj="$TD/$TPL/$NAME/summary.json"
  [ -f "$Sj" ] && "$PY" -c "import json;d=json.load(open('$Sj'));print(' $TPL score',d.get('score'),d.get('metric'),d.get('filter'))" 2>/dev/null
done
echo "compare vs v8-average: HE+ 93.29 / MPE 89.33 ; GPQA 99/198 (gap 0/63)"
echo "###### LSFOLD_EVAL_DONE $(ts) ######"
