#!/usr/bin/env bash
# lcb55_pub_v7_minprep.sh <gpu> [port] — LCB-55 on the PUBLISHED v7-coder Q6_K
# under the SAME served anti-loop sampler vendor_minp_rep @ t0.9 as the soft2
# run. This is the deployment-condition denominator: "did LCB degrade?" is only
# apples-to-apples when both ship candidate and published baseline are measured
# under the sampler they are actually served with. Pinned GPU (gated <2000 MiB).
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
MODEL=/mnt/sdc/ml/sft_heal/pub_v7_gguf/gemma-4-A4B-98e-v7-coder-it-Q6_K.gguf
TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
OUT=/srv/ml/agentic_loop/results/pub_v7_lcb55_minprep
NAME=pub-v7-q6-minprep09
TPL=lcb_medium_55_v4_minprep09
GPU=${1:?usage: lcb55_pub_v7_minprep.sh <gpu> [port]}; PORT=${2:-8193}
export LCB_TEMPERATURE=0.9
export LCB_TOP_P=0.95
export LCB_TOP_K=64
export LCB_MIN_P=0.05
export LCB_REPEAT_PENALTY=1.1
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
ts(){ date '+%T %Z'; }
for f in "$PY" "$OMK" "$MODEL" "$TOK/tokenizer.json" \
         "/srv/ml/repos/omnimergekit/eval/templates/${TPL}.yaml"; do
  [ -e "$f" ] || { echo "[lcb55-pub-minp] FATAL missing $f"; exit 9; }
done
U=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')
echo "[lcb55-pub-minp $(ts)] GPU$GPU used=${U} MiB sampler t=$LCB_TEMPERATURE minp=$LCB_MIN_P rep=$LCB_REPEAT_PENALTY"
[ "${U:-99999}" -lt 2000 ] || { echo "[lcb55-pub-minp] FATAL GPU$GPU busy (${U} MiB)"; exit 8; }
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
echo "[lcb55-pub-minp $(ts)] LCB55_PUB_MINP_DONE rc=$rc"
S="$OUT/${TPL}/$NAME/summary.json"
[ -f "$S" ] && "$PY" -c "import json;d=json.load(open('$S'));print('SCORE',d['score'],d['scores'],'finish',d['token_stats']['finish_reasons'],'charp50',d['token_stats']['completion_chars']['p50'])"
exit $rc
