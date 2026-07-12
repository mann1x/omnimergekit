#!/usr/bin/env bash
# dern_knob_variant.sh <label> <corpus> <anchor> <freq_exp> <gpu> <port>
# One DERN-knob variant: redist -> F16 -> Q6_K noimat -> loop gate.
# Base selected via env (default = BARE dern11, i.e. the published v7-coder student + C6v3lcb keep-meta):
#   STUDENT   : bf16 student dir to fold into       (default published v7-coder 98e)
#   KEEPMETA  : keep/drop meta matching STUDENT      (default C6v3lcb)
#   TAG       : output naming prefix                 (default dern11-knob)
# redist pinned to <gpu> via CUDA_VISIBLE_DEVICES (--device cuda:0 = that GPU); the gate takes the
# absolute <gpu> arg and pins itself, so we do NOT export CUDA_VISIBLE_DEVICES globally.
set -uo pipefail
LABEL=$1; CORPUS=$2; ANCHOR=$3; FEXP=$4; GPU=$5; PORT=$6
PY=/root/anaconda3/envs/omnimergekit/bin/python
DERN=/srv/ml/repos/omnimergekit/scripts/redist_dern_eq11.py
LCPP=/mnt/sdc/ml/llama.cpp-latest
SFT=/mnt/sdc/ml/sft_heal
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
STUDENT="${STUDENT:-/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-it}"
KEEPMETA="${KEEPMETA:-$SFT/v7coder_C6v3lcb_keepmeta.json}"
TAG="${TAG:-dern11-knob}"
GATE=/srv/ml/scripts/gate_sweep48_minp_p.sh
AL=/srv/ml/agentic_loop
OUT=$SFT/${TAG}-$LABEL-it
F16=$SFT/${TAG}-$LABEL-F16.gguf
Q6=$SFT/gemma-4-A4B-98e-v7-coder-${TAG}-$LABEL-it-Q6_K.gguf
RESULT=$AL/results/${TAG}-${LABEL}_minp48.json
ts(){ date '+%T %Z'; }
echo "[$TAG-$LABEL $(ts)] base=$(basename "$STUDENT") corpus=$(basename "$CORPUS") anchor=$ANCHOR freq_exp=$FEXP GPU=$GPU"

for f in "$STUDENT/config.json" "$KEEPMETA" "$CORPUS" "$DERN" "$GATE"; do
  [ -e "$f" ] || { echo "[$LABEL] FATAL missing $f"; exit 9; }
done

# 1. redist (patched DERN with knobs) on the chosen GPU
if [ ! -f "$OUT/model.safetensors.index.json" ] && [ ! -f "$OUT/model.safetensors" ]; then
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "$DERN" --teacher "$SRC" --student "$STUDENT" --keep-meta "$KEEPMETA" \
      --freq-corpus "$CORPUS" --seq-corpus "$CORPUS" --out "$OUT" --device cuda:0 \
      --norm-anchor "$ANCHOR" --freq-exponent "$FEXP" \
    || { echo "[$LABEL] FATAL redist"; exit 1; }
fi

# 2. convert F16
if [ ! -f "$F16" ]; then
  "$PY" "$LCPP/convert_hf_to_gguf.py" "$OUT" --outfile "$F16" --outtype f16 \
    || { echo "[$LABEL] FATAL convert"; exit 2; }
fi

# 3. quantize Q6_K (noimat — single-variable vs dern11-Q6_K-noimat baseline 4/48,2/48)
if [ ! -f "$Q6" ]; then
  "$LCPP/build/bin/llama-quantize" "$F16" "$Q6" Q6_K 32 \
    || { echo "[$LABEL] FATAL quant"; exit 3; }
fi

# 4. loop gate {0.9, 0.8}
echo "[$TAG-$LABEL $(ts)] gate {0.9,0.8} GPU$GPU:$PORT"
bash "$GATE" "$Q6" "$GPU" "$PORT" "$RESULT" "${TAG}-$LABEL"

# 5. drop F16 intermediate (keep bf16 dir + Q6 for winners)
rm -f "$F16"
echo "[$TAG-$LABEL $(ts)] DONE  Q6=$Q6"
