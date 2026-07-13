#!/bin/bash
# Build the LCB-targeted calibration corpus for the Qwen coder variant.
#
#   1. wait until GPU0 is free (no llama-server / omk lcb holding port 8091)
#   2. run 256e generation+scoring on the 103 disjoint LCB problems (lcb_calib template,
#      24k-generous recipe, MTP nextn) -> lcb_result.samples.jsonl with PASS/FAIL labels
#   3. harvest the PASS subset -> results/router_calib_corpus_lcb_qwen.jsonl (targeted_lcb)
#
# Runs the model's OWN passing full-CoT solutions through the router — the v7-coder
# 128e-PASS targeted channel. Launch detached (setsid) after the current eval; it self-gates
# on GPU0 so it is safe to fire immediately.
set -u
cd /srv/ml/repos/omnimergekit
export HF_HUB_ENABLE_HF_TRANSFER=0 CUDA_VISIBLE_DEVICES=0
export LM_EVAL_BIN=/root/anaconda3/envs/omnimergekit/bin/lm-eval
PY=/root/anaconda3/envs/omnimergekit/bin/python
RES=/srv/ml/eval_results/qwen_calib
G256=/srv/ml/models/gguf/Qwen3.6-35B-A3B-256e-GGUF/Qwen3.6-35B-A3B-Q6_K.gguf
T256=/srv/ml/models/Qwen3.6-35B-A3B
PORT=8091
NAME=qwen256e_q6k

# double-launch guard
if pgrep -f "omk_eval.py --backend llama --template lcb_calib" >/dev/null; then
  echo "!!!! lcb_calib generation already running — abort"; exit 1
fi

echo ">>>> $(date -Iseconds) waiting for GPU0/port $PORT free ..."
while pgrep -f "omk_eval.py --backend llama --template lcb" >/dev/null \
      || ss -ltn 2>/dev/null | grep -q ":$PORT "; do
  sleep 60
done
echo ">>>> $(date -Iseconds) GPU0 free — starting 256e LCB calib generation (103 problems, 24k)"

export LLAMA_ARG_SPEC_TYPE=draft-mtp
$PY eval/omk_eval.py --backend llama --template lcb_calib --quant q6_k \
  --model "$G256" --tokenizer "$T256" --served-name "$NAME" --port "$PORT" \
  --results-dir "$RES" --parallel 2
GEN_RC=$?
echo "<<<< $(date -Iseconds) generation rc=$GEN_RC"

SAMPLES=$RES/lcb_calib/$NAME/lcb_result.samples.jsonl
if [ ! -s "$SAMPLES" ]; then
  echo "!!!! no samples at $SAMPLES — generation failed, no corpus written"; exit 2
fi
echo ">>>> $(date -Iseconds) harvesting PASS -> targeted_lcb corpus"
$PY recipes/qwen3_6_35b_a3b_prune/harvest_lcb_calib_corpus.py \
  --samples "$SAMPLES" --tokenizer "$T256" \
  --out recipes/qwen3_6_35b_a3b_prune/results/router_calib_corpus_lcb_qwen.jsonl
echo "==== LCB CALIB CORPUS DONE $(date -Iseconds) ===="
