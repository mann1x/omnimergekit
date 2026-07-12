#!/usr/bin/env bash
# lcb55_v8_fkbroad_minprep.sh [PORT] [GPU] — LCB-55 on the v8 fkbroad-soft2
# imat-Q6 candidate under the SERVED anti-loop sampler vendor_minp_rep @ t0.9
# (the 0/48 agentic-gate config). Apples-to-apples with lcb55_soft2_minprep.sh
# and lcb55_pub_v7_minprep.sh: same omk, same lcb_medium_55_v4_minprep09
# template, same LCB_* env sampler, same tokenizer, --parallel 2. Pinned to a
# GPU arg (default 1) to run concurrently with Part C on the vendor arm; the
# greedy b9700 anchor on GPU0 is left untouched.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
MODEL=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-fkbroad-soft2-imat-Q6_K.gguf
TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
OUT=/srv/ml/agentic_loop/results/fkbroad_soft2_lcb55_minprep
NAME=fkbroad-soft2-imat-minprep09
TPL=lcb_medium_55_v4_minprep09
PORT=${1:-8192}
GPU=${2:-1}
# vendor_minp_rep @ t0.9 — the 0/48 agentic-gate config
export LCB_TEMPERATURE=0.9
export LCB_TOP_P=0.95
export LCB_TOP_K=64
export LCB_MIN_P=0.05
export LCB_REPEAT_PENALTY=1.1
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
ts(){ date '+%T %Z'; }
echo "[lcb55-v8-minp $(ts)] sampler: t=$LCB_TEMPERATURE top_p=$LCB_TOP_P top_k=$LCB_TOP_K min_p=$LCB_MIN_P rep=$LCB_REPEAT_PENALTY"
for f in "$PY" "$OMK" "$MODEL" "$TOK/tokenizer.json" \
         "/srv/ml/repos/omnimergekit/eval/templates/${TPL}.yaml"; do
  [ -e "$f" ] || { echo "[lcb55-v8-minp] FATAL missing $f"; exit 9; }
done
echo "[lcb55-v8-minp $(ts)] launching on GPU$GPU PORT=$PORT"
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
echo "[lcb55-v8-minp $(ts)] LCB55_MINP_DONE rc=$rc"
S="$OUT/${TPL}/$NAME/summary.json"
[ -f "$S" ] && "$PY" -c "import json;d=json.load(open('$S'));print('SCORE',d['score'],d.get('scores'),'finish',d['token_stats']['finish_reasons'],'charp50',d['token_stats']['completion_chars']['p50'])"
exit $rc
