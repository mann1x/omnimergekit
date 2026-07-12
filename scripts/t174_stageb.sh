#!/bin/bash
# T174 Stage-B: merge epoch2 adapter -> Q6_K (A2 imatrix) -> dual-GPU 3-bench
# audit vs A2. Mirrors t172_comboA_dual eval_stream (one model per GPU, distinct
# ports, explicit-pid wait — no tee-coproc deadlock).
set -uo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH   # lm-eval CLI must resolve (v1 stageb FileNotFoundError fix)
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
AUDIT=/srv/ml/repos/omnimergekit/scripts/audit_full_bench.py
A2=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it
ADAPTER=${1:-/mnt/sdc/ml/google/a2-antiloop-lora/epoch2}
NAME=${2:-a2-antiloop-e2}
MERGED=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-${NAME}-it
GGUF=$MERGED-GGUF
F16=$GGUF/$NAME-F16.gguf
Q6=$GGUF/$NAME-Q6_K.gguf
IMAT=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it-GGUF/imatrix.dat
CONVERT=/opt/llama.cpp/convert_hf_to_gguf.py
QUANT=/opt/llama.cpp/build/bin/llama-quantize
RES=/srv/ml/eval_results_tracks_2_3
BASE=a2-62e-fc15_25-p8-s1_0p1_20

echo "==================== t174_stageb $NAME $(date -Iseconds) ===================="
[ -f "$IMAT" ] || { echo "FATAL: A2 imatrix missing"; exit 2; }

# 1 merge
if [ ! -f "$MERGED/model.safetensors.index.json" ]; then
  echo "[merge $(date +%H:%M:%S)] $ADAPTER -> $MERGED"
  CUDA_VISIBLE_DEVICES=0 $PY /srv/ml/scripts/merge_adapter.py "$A2" "$ADAPTER" "$MERGED" || { echo "FATAL merge"; exit 3; }
fi
# 2 F16
mkdir -p "$GGUF"
if [ ! -f "$F16" ] && [ ! -f "$Q6" ]; then
  echo "[f16 $(date +%H:%M:%S)] convert"
  $PY "$CONVERT" "$MERGED" --outfile "$F16" --outtype f16 2>&1 | tail -6 || { echo "FATAL convert"; exit 4; }
fi
# 3 Q6_K with A2 imatrix
if [ ! -f "$Q6" ]; then
  echo "[q6 $(date +%H:%M:%S)] quantize (A2 imatrix)"
  "$QUANT" --imatrix "$IMAT" "$F16" "$Q6" Q6_K 2>&1 | tail -6 || { echo "FATAL quant"; exit 5; }
fi
[ -f "$Q6" ] || { echo "FATAL: Q6_K missing"; exit 6; }
[ -f "$F16" ] && rm -f "$F16"   # reclaim ~28 GB
echo "[built $(date +%H:%M:%S)] $Q6 ($(du -h "$Q6" | cut -f1))"

# 4 dual-GPU eval (GPU0: HE+ + ifeval ; GPU1: MPE)
eval_stream(){ local gpu=$1 port=$2; shift 2; local tpl sd tlim; for tpl in "$@"; do
  sd=$RES/$tpl/$NAME; [ -f "$sd/summary.json" ] && { echo "[g$gpu] SKIP $tpl"; continue; }
  tlim=2400; [ "$tpl" = multipl_e_100 ] && tlim=3600
  pkill -KILL -f "llama-server.*--port $port" 2>/dev/null; sleep 2
  echo "[g$gpu $(date +%H:%M:%S)] eval $tpl"
  CUDA_VISIBLE_DEVICES=$gpu timeout --kill-after=10 --signal=KILL $tlim $PY $OMK \
    --model "$Q6" --tokenizer "$MERGED" --template $tpl --backend llama \
    --port $port --served-name $NAME --results-dir $RES 2>&1 | sed "s/^/[g$gpu] /" | tail -5
  pkill -KILL -f "llama-server.*--port $port" 2>/dev/null; sleep 2
done; echo "[g$gpu] STREAM DONE"; }

echo "==================== t174_stageb EVAL $(date -Iseconds) ===================="
eval_stream 0 8195 humanevalplus_full ifeval_100 & p0=$!
eval_stream 1 8295 multipl_e_100 & p1=$!
wait "$p0" "$p1"

# 5 audit vs A2
echo "==================== t174_stageb AUDIT vs $BASE ===================="
for tpl in humanevalplus_full ifeval_100 multipl_e_100; do
  OMK_AUDIT_ROOT=$RES $PY $AUDIT $tpl "$NAME" "$BASE" 2>/dev/null | grep "^AUDIT" || echo "AUDIT_FAIL $tpl"
done
echo "==================== t174_stageb DONE $(date +%H:%M:%S) ===================="
