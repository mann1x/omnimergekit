#!/usr/bin/env bash
# GPU0: v6-coder Q6_K (HF download) -> HE+164 + LCB-55 on bs2 (same binary as v7 runs)
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
export HF_XET_HIGH_PERFORMANCE=1
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH
PY=/srv/ml/envs/envs/omnimergekit/bin/python
HF=/srv/ml/envs/envs/omnimergekit/bin/hf
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
GGDIR=/mnt/sdc/ml/google/gemma-4-A4B-98e-v6-coder-it-GGUF
Q6="$GGDIR/gemma-4-A4B-98e-v6-coder-it-Q6_K.gguf"
TOK=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-floor-it
RES=/srv/ml/eval_results_v6coder_bs2
NAME=v6coder-q6k
PORT=8195
L(){ echo "[v6code $(date -u +%H:%M:%S)] $*"; }
mkdir -p "$GGDIR" "$RES"
if [ ! -f "$Q6" ]; then
  L "downloading Q6_K gguf..."
  "$HF" download ManniX-ITA/gemma-4-A4B-98e-v6-coder-it-GGUF gemma-4-A4B-98e-v6-coder-it-Q6_K.gguf --local-dir "$GGDIR" || { L "DL FAIL"; exit 1; }
fi
ls -la "$Q6"
sc(){ "$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get(\"score\"))" "$1" 2>/dev/null; }
for tpl in humanevalplus_full lcb_medium_55_v4; do
  L ">>> eval $tpl (llama backend, GPU0, port $PORT)"
  "$PY" "$OMK" --model "$Q6" --tokenizer "$TOK" --template "$tpl" --backend llama --port "$PORT" --served-name "$NAME" --results-dir "$RES" 2>&1 | tail -20
  L "SCORE $tpl = $(sc "$RES/$tpl/$NAME/summary.json")"
done
L "V6_CODE_DONE he+=$(sc "$RES/humanevalplus_full/$NAME/summary.json") lcb=$(sc "$RES/lcb_medium_55_v4/$NAME/summary.json")"
