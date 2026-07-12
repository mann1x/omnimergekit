#!/usr/bin/env bash
# lcb55_pub_v7.sh [GPU] [PORT] — same-stack LCB-55 control on the PUBLISHED
# v7-coder Q6_K GGUF, on THIS bs2 llama.cpp (sm_120) rig. Decisive anchor for
# whether the soft2 LCB-55 = 0.8909 (49/55) is a real regression vs the card's
# 96.36% (53/55) or just a host/llama.cpp-build difference. Template, sampler
# (frozen greedy), thinking budget, tokenizer, parallelism all mirror the soft2
# run exactly (results layout soft2_imat_lcb55) — only the model + served-name
# + outdir differ, so the two summary.json scores are apples-to-apples.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
MODEL=/mnt/sdc/ml/sft_heal/pub_v7_gguf/gemma-4-A4B-98e-v7-coder-it-Q6_K.gguf
TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
OUT=/srv/ml/agentic_loop/results/pub_v7_lcb55
NAME=pub-v7-q6
GPU=${1:-0}; PORT=${2:-8190}
ts(){ date '+%T %Z'; }
echo "[lcb55-pub $(ts)] GPU=$GPU PORT=$PORT model=$(basename "$MODEL")"
for f in "$PY" "$OMK" "$MODEL" "$TOK/tokenizer.json"; do
  [ -e "$f" ] || { echo "[lcb55-pub] FATAL missing $f"; exit 9; }
done
# GPU-memory preflight (launch only when target GPU is idle, <2000 MiB)
USED=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')
echo "[lcb55-pub $(ts)] GPU$GPU used=${USED} MiB"
[ "${USED:-99999}" -lt 2000 ] || { echo "[lcb55-pub] FATAL GPU$GPU busy (${USED} MiB) — refuse to launch"; exit 8; }
mkdir -p "$OUT"
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
CUDA_VISIBLE_DEVICES=$GPU "$PY" "$OMK" \
  --model "$MODEL" \
  --template lcb_medium_55_v4 \
  --backend llama \
  --quant gguf \
  --port "$PORT" \
  --results-dir "$OUT" \
  --served-name "$NAME" \
  --tokenizer "$TOK" \
  --parallel 2
rc=$?
echo "[lcb55-pub $(ts)] LCB55_PUB_DONE rc=$rc"
[ -f "$OUT/lcb_medium_55_v4/$NAME/summary.json" ] && \
  "$PY" -c "import json;d=json.load(open('$OUT/lcb_medium_55_v4/$NAME/summary.json'));print('SCORE',d['score'],d['scores'])"
exit $rc
