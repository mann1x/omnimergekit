#!/usr/bin/env bash
# lcb55_soft2_minprep.sh [PORT] — LCB-55 on the soft2 imat-Q6 ship candidate
# under the SERVED anti-loop sampler vendor_minp_rep @ temp 0.9 (the exact
# config that gave 0/48 on the 48-seed agentic gate). Self-scheduling: waits
# for the first free GPU (<2000 MiB), pins it, runs through the normal omk
# pipeline (patched env-driven LCB sampler). Greedy canonical path untouched.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
MODEL=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-dern11-soft-soft2-imat-Q6_K.gguf
TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
OUT=/srv/ml/agentic_loop/results/soft2_imat_lcb55_minprep
NAME=dern11-soft-soft2-imat-minprep09
TPL=lcb_medium_55_v4_minprep09
PORT=${1:-8192}
# vendor_minp_rep @ t0.9 — the 0/48 agentic-gate config
export LCB_TEMPERATURE=0.9
export LCB_TOP_P=0.95
export LCB_TOP_K=64
export LCB_MIN_P=0.05
export LCB_REPEAT_PENALTY=1.1
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
ts(){ date '+%T %Z'; }
echo "[lcb55-minp $(ts)] sampler: t=$LCB_TEMPERATURE top_p=$LCB_TOP_P top_k=$LCB_TOP_K min_p=$LCB_MIN_P rep=$LCB_REPEAT_PENALTY"
for f in "$PY" "$OMK" "$MODEL" "$TOK/tokenizer.json" \
         "/srv/ml/repos/omnimergekit/eval/templates/${TPL}.yaml"; do
  [ -e "$f" ] || { echo "[lcb55-minp] FATAL missing $f"; exit 9; }
done
# self-schedule: wait up to 4h for the first GPU <2000 MiB
GPU=""
for i in $(seq 1 240); do
  for g in 0 1; do
    U=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')
    if [ "${U:-99999}" -lt 2000 ]; then GPU=$g; break; fi
  done
  [ -n "$GPU" ] && break
  echo "[lcb55-minp $(ts)] both GPUs busy (g0+g1 >=2000 MiB), wait 60s ($i/240)"
  sleep 60
done
[ -n "$GPU" ] || { echo "[lcb55-minp] FATAL no free GPU after 4h"; exit 8; }
echo "[lcb55-minp $(ts)] launching on GPU$GPU PORT=$PORT"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU "$PY" "$OMK" \
  --model "$MODEL" \
  --template "$TPL" \
  --backend llama \
  --quant gguf \
  --port "$PORT" \
  --results-dir "$OUT" \
  --served-name "$NAME" \
  --tokenizer "$TOK" \
  --parallel 2
rc=$?
echo "[lcb55-minp $(ts)] LCB55_MINP_DONE rc=$rc"
S="$OUT/${TPL}/$NAME/summary.json"
[ -f "$S" ] && "$PY" -c "import json;d=json.load(open('$S'));print('SCORE',d['score'],d['scores'],'finish',d['token_stats']['finish_reasons'],'charp50',d['token_stats']['completion_chars']['p50'])"
exit $rc
