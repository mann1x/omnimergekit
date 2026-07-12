#!/usr/bin/env bash
# reg_q40_code_eval.sh — matched CODE control for the qat-Q4_0 attribution.
# Evals the regular-DERN base, NO-imatrix Q4_0 GGUF (built by reg_q40_control.sh) on
# HE+164 + MPE-100, b9700 eval binary, greedy, --parallel 2 — identical to the qat-Q4_0
# step-8 eval. This is the MATCHED comparison qat-Q4_0 (81.10 / 67.67) needs: same
# no-imatrix Q4_0 quant, regular vs QAT base. If this lands ~88-92 / ~84-88 then the
# qat code collapse is real QAT-cooking damage; if it lands ~81 / ~68 too then it is the
# no-imatrix-Q4_0 quant, not the base. Runs on GPU0 (loop gate holds GPU1). PID-kill only.
set -uo pipefail
GGUF=/mnt/sdc/ml/v8_qat/reg_ctrl/gemma-4-A4B-98e-v7-coder-reg-Q4_0.gguf
TD=/mnt/sdc/ml/v8_qat/reg_ctrl/results
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
LLAMA_BIN_EVAL=/mnt/sdc/ml/llama.cpp-b9700/build/bin
BASE_TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
mkdir -p "$TD"
ts(){ date '+%T %Z'; }
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
export HF_ALLOW_CODE_EVAL=1
echo "==== reg-Q4_0 (no-imat) CODE eval $(ts) ===="
# wait until the control GGUF exists (loop gate's build writes it first)
for i in $(seq 1 120); do [ -f "$GGUF" ] && break; echo "[wait $(ts)] gguf not ready ($i)"; sleep 5; done
[ -f "$GGUF" ] || { echo "FATAL gguf missing $GGUF"; exit 2; }
p=8295
for TPL in humanevalplus_full multipl_e_100; do
  echo "[eval $(ts)] $TPL (GPU0:$p)"
  LLAMA_BIN="$LLAMA_BIN_EVAL" CUDA_VISIBLE_DEVICES=0 "$PY" "$OMK" \
    --model "$GGUF" --template "$TPL" --backend llama --quant gguf \
    --port "$p" --results-dir "$TD" --served-name v8-reg-Q4_0 \
    --tokenizer "$BASE_TOK" --parallel 2 || echo "[eval $(ts)] WARN $TPL rc=$?"
  p=$((p+1))
done
echo "[done $(ts)] ===== reg-Q4_0 CODE results ====="
for TPL in humanevalplus_full multipl_e_100; do
  S="$TD/$TPL/v8-reg-Q4_0/summary.json"
  [ -f "$S" ] && grep -oE '"score"[: ]+[0-9.]+' "$S" | head -1 | sed "s/^/  $TPL /"
done
echo "###### REG_Q40_CODE_DONE $(ts) ######"
