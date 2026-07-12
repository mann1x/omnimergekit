#!/usr/bin/env bash
# run_coderx_v6_55_chain.sh — when GPU1's LCB55 (code4/lcb3) finishes (server torn
# down → GPU1 free), fire lcb_v6_55 on the already-built coderx-STD16 imat-Q6.
# No rebuild: CX16c4l3-imat-Q6_K.gguf already exists.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
PYE=/srv/ml/envs/envs/omnimergekit/bin/python
BIN=/srv/ml/envs/envs/omnimergekit/bin
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
CXLCB55=/srv/ml/eval_results_cx_std16/lcb_medium_55_v4/cx16-c4l3-imatq6/summary.json
Q6=/mnt/sdc/ml/cx_std16/CX16c4l3-imat-Q6_K.gguf
TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
RES=/srv/ml/eval_results_lcb_v6
LOG=/mnt/sdc/ml/cx_std16/lcb_v6_55_coderx.log
ts(){ date '+%T %Z'; }

echo "[chain $(ts)] waiting for GPU1 LCB55 (code4/lcb3) summary → server-down ..."
for _ in $(seq 1 60); do [ -f "$CXLCB55" ] && break; sleep 30; done
if [ -f "$CXLCB55" ]; then
  echo "[chain $(ts)] LCB55 code4/lcb3 score=$("$PY" -c "import json;print(json.load(open('$CXLCB55')).get('score'))")"
else
  echo "[chain $(ts)] WARN LCB55 summary missing after 30m; proceeding"
fi
sleep 12   # let GPU1 llama-server fully release VRAM

echo "[chain $(ts)] launching lcb_v6_55 on coderx-STD16 (code4/lcb3) Q6 — GPU1:8415"
cd /srv/ml/repos/omnimergekit
env PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES=1 \
  "$PYE" "$OMK" --model "$Q6" --tokenizer "$TOK" --template lcb_v6_55 \
  --backend llama --port 8415 --served-name cx16-c4l3 --results-dir "$RES" \
  > "$LOG" 2>&1 || echo "[chain] WARN omk rc=$?"

S=$RES/lcb_v6_55/cx16-c4l3/summary.json
if [ -f "$S" ]; then
  echo "[chain $(ts)] lcb_v6_55 coderx-STD16 score=$("$PY" -c "import json;print(json.load(open('$S')).get('score'))")"
else
  echo "[chain $(ts)] WARN coderx v6-55 summary missing; tail:"; tail -4 "$LOG"
fi
echo "###### CODERX_V6_55_CHAIN_DONE $(ts) ######"
