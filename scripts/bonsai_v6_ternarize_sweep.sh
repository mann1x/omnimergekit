#!/usr/bin/env bash
# bonsai_v6_ternarize_sweep.sh — which training-free ternarizer is least bad end to end?
#
# Input: the already-folded v6 bf16 GGUF (PrismML contract, MTP head grafted unfolded).
# Each arm ternarizes the 402 contract tensors with one rule, leaves everything else alone, and
# is measured as a bf16 GGUF on the prism fork against the fork's own bf16 base logits. The
# emitted values are exact +-d, so the PQ2_0/PTQ1_0 codec packs them losslessly: the number
# measured here IS the number the packed file will have (re-checked on the winner).
# One 54 G arm file exists at a time; it is deleted after measurement.
set -uo pipefail
ts(){ date -u '+%H:%M:%SZ'; }
PY=/root/anaconda3/envs/omnimergekit/bin/python
export PYTHONPATH=/srv/ml/dca-cleanroom/llama.cpp/gguf-py
PRISM=/srv/ml/bonsai-assess/ref-b10754/llama-prism-b10754-2459f68
DIR=/mnt/sdc/opencoti-models/sdc-ml/omnimerge-v6
SRC=$DIR/Qwen3.8-27B-Omnimerge-v6-BF16-folded-MTP.gguf
REF=/mnt/sdc/opencoti-models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf
OUT=/mnt/sdc/opencoti-scratch/v6-ternary/r2; mkdir -p "$OUT"
BASE=/mnt/sdc/opencoti-scratch/v6-ternary/bf16-c2048-prism.kld
TEXT=/srv/ml/ppl-corpus/wiki.test.mid.raw
[ -s "$SRC" ] && [ -s "$BASE" ] || { echo "FATAL missing input"; exit 2; }
echo "===== V6 TERNARIZER SWEEP $(ts) ====="
for MODE in "$@"; do
  TAG=${MODE//:/_}; F=$DIR/v6-tern-$TAG.gguf
  echo "[$(ts)] arm $MODE"
  "$PY" /srv/ml/scripts/bonsai_hadamard_fold.py --src "$SRC" --ref "$REF" --ternarize "$MODE" --out "$F" --device cuda:0 > "$OUT/tern-$TAG.log" 2>&1
  grep -q FOLD_OK "$OUT/tern-$TAG.log" || { echo "   FATAL ternarize"; tail -8 "$OUT/tern-$TAG.log"; rm -f "$F"; continue; }
  LD_LIBRARY_PATH="$PRISM" CUDA_VISIBLE_DEVICES=0 "$PRISM/llama-perplexity" -m "$F" -f "$TEXT" \
     --kl-divergence-base "$BASE" --kl-divergence -c 2048 --chunks 8 -b 2048 -ub 512 -ngl 99 > "$OUT/kld-$TAG.log" 2>&1 \
     || { echo "   FAIL kld"; tail -6 "$OUT/kld-$TAG.log"; }
  grep -E "^(Mean  *PPL\(Q\) |Mean  *KLD|Same top p)" "$OUT/kld-$TAG.log" | sed 's/^/     /'
  rm -f "$F"
done
echo "[$(ts)] === SWEEP_DONE ==="
