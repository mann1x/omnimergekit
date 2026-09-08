#!/bin/bash
# Build the MPE-targeted calibration corpus for the Qwen coder LCB+MPE variant.
#   1. wait for GPU0/port 8091 free
#   2. 256e teacher generates+Docker-scores the 175 DISJOINT MultiPL-E problems
#      (index>=100 per rs/java/js; multipl_e_calib template, chat mode, temp 0.6,
#      fixed chat template, MTP nextn) -> per-problem PASS labels
#   3. harvest PASS -> results/router_calib_corpus_mpe_qwen.jsonl (targeted_mpe)
set -u
cd /srv/ml/repos/omnimergekit
export HF_HUB_ENABLE_HF_TRANSFER=0 CUDA_VISIBLE_DEVICES=0
export LM_EVAL_BIN=/root/anaconda3/envs/omnimergekit/bin/lm-eval
export LLAMA_ARG_SPEC_TYPE=draft-mtp
export LLAMA_EXTRA="--jinja --chat-template-file /srv/ml/models/qwen36_chat_template_fixed.jinja"
PY=/root/anaconda3/envs/omnimergekit/bin/python
RES=/srv/ml/eval_results/qwen_calib
G256=/srv/ml/models/gguf/Qwen3.6-35B-A3B-256e-GGUF/Qwen3.6-35B-A3B-Q6_K.gguf
T256=/srv/ml/models/Qwen3.6-35B-A3B
PORT=8091; NAME=qwen256e_q6k
if pgrep -f "omk_eval.py --backend llama --template multipl_e_calib" >/dev/null; then
  echo "!!!! mpe_calib generation already running — abort"; exit 1; fi
echo ">>>> $(date -Iseconds) waiting for GPU0/port $PORT free ..."
while ss -ltn 2>/dev/null | grep -q ":$PORT "; do sleep 30; done
echo ">>>> $(date -Iseconds) GPU0 free — 256e MPE calib gen (175 disjoint, chat temp0.6)"
$PY eval/omk_eval.py --backend llama --template multipl_e_calib --quant q6_k \
  --model "$G256" --tokenizer "$T256" --served-name "$NAME" --port "$PORT" \
  --results-dir "$RES" --parallel 2 --sampler-profile qwen3_6 --sampler recommended
GRC=$?
echo "<<<< $(date -Iseconds) mpe calib gen+eval rc=$GRC"
OUT=$RES/multipl_e_calib/$NAME
if [ ! -d "$OUT/generations" ]; then echo "!!!! no generations at $OUT — abort"; exit 2; fi
echo ">>>> $(date -Iseconds) harvest PASS -> targeted_mpe corpus"
$PY recipes/qwen3_6_35b_a3b_prune/harvest_mpe_calib_corpus.py \
  --out-dir "$OUT" --tokenizer "$T256" \
  --out recipes/qwen3_6_35b_a3b_prune/results/router_calib_corpus_mpe_qwen.jsonl
echo "==== MPE CALIB CORPUS DONE $(date -Iseconds) ===="
