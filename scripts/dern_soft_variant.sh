#!/usr/bin/env bash
# dern_soft_variant.sh <label> <topk> <gpu> <port>
# Soft top-k assignment DERN variant on BARE dern11 (published v7-coder 98e + C6v3lcb keep-meta):
#   redist --assign-topk <topk> -> F16 -> Q6_K noimat -> loop gate vendor_minp_rep {0.9,0.8}.
# Single variable vs baseline dern11 (hard argmax = 4/48@t0.9, 2/48@t0.8). Predicted regression
# (an-mean 14/48 + fx-0.5 19/48 showed the fold wants mass concentrated, not spread) — this is the
# deliberate falsification of that read.
set -uo pipefail
LABEL=$1; TOPK=$2; GPU=$3; PORT=$4
PY=/root/anaconda3/envs/omnimergekit/bin/python
DERN=/srv/ml/repos/omnimergekit/scripts/redist_dern_eq11.py
LCPP=/mnt/sdc/ml/llama.cpp-latest
SFT=/mnt/sdc/ml/sft_heal
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
STUDENT=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-it
KEEPMETA=$SFT/v7coder_C6v3lcb_keepmeta.json
CORPUS=$SFT/eog_corpus_solar.jsonl
GATE=/srv/ml/scripts/gate_sweep48_minp_p.sh
AL=/srv/ml/agentic_loop
OUT=$SFT/dern11-soft-$LABEL-it
F16=$SFT/dern11-soft-$LABEL-F16.gguf
Q6=$SFT/gemma-4-A4B-98e-v7-coder-dern11-soft-$LABEL-it-Q6_K.gguf
RESULT=$AL/results/dern11-soft-${LABEL}_minp48.json
ts(){ date '+%T %Z'; }
echo "[soft-$LABEL $(ts)] base=$(basename "$STUDENT") assign_topk=$TOPK GPU=$GPU"
for f in "$STUDENT/config.json" "$KEEPMETA" "$CORPUS" "$DERN" "$GATE"; do
  [ -e "$f" ] || { echo "[$LABEL] FATAL missing $f"; exit 9; }
done
# preflight: confirm the redist supports --assign-topk (patched)
grep -q -- "--assign-topk" "$DERN" || { echo "[$LABEL] FATAL redist not patched (no --assign-topk)"; exit 8; }

# 1. redist soft top-k on chosen GPU
if [ ! -f "$OUT/model.safetensors" ] && [ ! -f "$OUT/model.safetensors.index.json" ]; then
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "$DERN" --teacher "$SRC" --student "$STUDENT" --keep-meta "$KEEPMETA" \
      --freq-corpus "$CORPUS" --seq-corpus "$CORPUS" --out "$OUT" --device cuda:0 \
      --assign-topk "$TOPK" \
    || { echo "[$LABEL] FATAL redist"; exit 1; }
fi
# 2. F16
[ -f "$F16" ] || "$PY" "$LCPP/convert_hf_to_gguf.py" "$OUT" --outfile "$F16" --outtype f16 \
  || { echo "[$LABEL] FATAL convert"; exit 2; }
# 3. Q6 noimat
[ -f "$Q6" ] || "$LCPP/build/bin/llama-quantize" "$F16" "$Q6" Q6_K 32 \
  || { echo "[$LABEL] FATAL quant"; exit 3; }
# 4. loop gate
echo "[soft-$LABEL $(ts)] gate {0.9,0.8} GPU$GPU:$PORT vs baseline 4/48,2/48"
bash "$GATE" "$Q6" "$GPU" "$PORT" "$RESULT" "dern11-soft-$LABEL"
# 5. drop F16 (keep bf16 + Q6 for a possible imat promote)
rm -f "$F16"
echo "[soft-$LABEL $(ts)] DONE  Q6=$Q6"
