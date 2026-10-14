#!/usr/bin/env bash
# bonsai_base_vs_prism.sh — the like-for-like comparison: base Qwen3.8-27B is the teacher for BOTH
# PrismML's shipped Bonsai 2 and our training-free ternarizations of that same base.
#   1. wait for the base safetensors download, convert to a bf16 GGUF
#   2. base logits on the prism fork
#   3. KLD of PrismML's own PQ2_0 against those logits   <- the target number
#   4. per arm: fold + ternarize the base in one pass, KLD against the same logits
# Arms are bf16 GGUFs holding exact +-d values (codec-lossless), one on disk at a time.
set -uo pipefail
ts(){ date -u '+%H:%M:%SZ'; }
PY=/root/anaconda3/envs/omnimergekit/bin/python
C=/srv/ml/dca-cleanroom/llama.cpp
export PYTHONPATH=$C/gguf-py
PRISM=/srv/ml/bonsai-assess/ref-b10754/llama-prism-b10754-2459f68
HF=/mnt/sdc/opencoti-models/hf/Qwen/Qwen3.8-27B
DIR=/mnt/sdc/opencoti-models/sdc-ml/qwen38-base; mkdir -p "$DIR"
GG=$DIR/Qwen3.8-27B-BF16.gguf
REF=/mnt/sdc/opencoti-models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf
OUT=/mnt/sdc/opencoti-scratch/v6-ternary/base; mkdir -p "$OUT"
BASE=$OUT/qwen38-bf16-c2048-prism.kld
TEXT=/srv/ml/ppl-corpus/wiki.test.mid.raw
ppl(){ LD_LIBRARY_PATH="$PRISM" CUDA_VISIBLE_DEVICES=0 "$PRISM/llama-perplexity" "$@" -f "$TEXT" -c 2048 --chunks 8 -b 2048 -ub 512 -ngl 99; }
show(){ grep -E "^(Final estimate|Mean  *PPL\(Q\) |Mean  *KLD|Same top p)" "$1" | sed 's/^/     /'; }
echo "===== BASE vs PRISM $(ts) ====="
i=0
until [ "$(ls "$HF"/model-000*-of-00018.safetensors 2>/dev/null | wc -l)" = 18 ] && ! ls "$HF"/*.aria2 >/dev/null 2>&1; do
  i=$((i+1)); [ $i -ge 240 ] && { echo "FATAL download did not finish in 60 min"; exit 1; }; sleep 15
done
echo "[1 $(ts)] download complete: $(du -sh "$HF" | cut -f1)"
if [ ! -s "$GG" ]; then
  "$PY" $C/convert_hf_to_gguf.py "$HF" --outtype bf16 --outfile "$GG" > "$OUT/convert.log" 2>&1 || { echo "FATAL convert"; tail -8 "$OUT/convert.log"; exit 1; }
fi
echo "[1 $(ts)] gguf $(stat -c%s "$GG") B"
[ -s "$BASE" ] || ppl -m "$GG" --kl-divergence-base "$BASE" > "$OUT/base.log" 2>&1 || { echo "FATAL base logits"; tail -8 "$OUT/base.log"; exit 1; }
echo "[2 $(ts)] base:"; show "$OUT/base.log"
echo "[3 $(ts)] PrismML Bonsai 2 PQ2_0 vs base Qwen3.8-27B"
ppl -m "$REF" --kl-divergence-base "$BASE" --kl-divergence > "$OUT/kld-prism.log" 2>&1 || tail -6 "$OUT/kld-prism.log"; show "$OUT/kld-prism.log"
for MODE in "$@"; do
  TAG=${MODE//:/_}; F=$DIR/qwen38-tern-$TAG.gguf
  echo "[4 $(ts)] arm $MODE"
  "$PY" /srv/ml/scripts/bonsai_hadamard_fold.py --src "$GG" --ref "$REF" --ternarize "$MODE" --out "$F" --device cuda:0 > "$OUT/tern-$TAG.log" 2>&1
  grep -q FOLD_OK "$OUT/tern-$TAG.log" || { echo "   FATAL"; tail -6 "$OUT/tern-$TAG.log"; rm -f "$F"; continue; }
  ppl -m "$F" --kl-divergence-base "$BASE" --kl-divergence > "$OUT/kld-$TAG.log" 2>&1 || tail -6 "$OUT/kld-$TAG.log"; show "$OUT/kld-$TAG.log"
  rm -f "$F"
done
echo "[$(ts)] === BASE_VS_PRISM_DONE ==="
