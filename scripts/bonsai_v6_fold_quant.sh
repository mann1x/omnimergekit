#!/usr/bin/env bash
# bonsai_v6_fold_quant.sh — re-quantize Omnimerge v6 to PQ2_0 / PTQ1_0 the way Bonsai 2 is laid out.
#
# The first v6 ternary files (2026-10-09 11:49) were round-to-nearest in the RAW basis with our
# quantizer's default type map. Three things differed from PrismML's files and all three are
# fixed here:
#   1. no Hadamard fold            -> fold first (bonsai_hadamard_fold.py), contract copied from
#                                     PrismML's own GGUF (same S, block 1024, GDN grouping)
#   2. ssm_alpha/ssm_beta ternary  -> kept BF16 (PrismML keeps exactly these 96)
#   3. token_embd Q4_K, output Q6_K-> both ternary, as PrismML has them
# What this does NOT add is recovery training. It makes the FORMAT match; quality is measured.
#
# Gates, in order, each fatal:
#   G1 LOSSLESS  the folded full-precision GGUF must reproduce the bf16 PPL on the prism fork.
#                Fails if the sign/permutation/axis convention is wrong — before any rounding.
#   G2 CENSUS    per-tensor type table identical to PrismML's file of the same packing.
#   G3 KV        prism.hadamard.* identical to PrismML's.
# Then KLD vs the fork's own bf16 base logits, and PPL, for both packings.
set -uo pipefail
ts(){ date -u '+%H:%M:%SZ'; }
PY=/root/anaconda3/envs/omnimergekit/bin/python
export PYTHONPATH=/srv/ml/dca-cleanroom/llama.cpp/gguf-py
PRISM=/srv/ml/bonsai-assess/ref-b10754/llama-prism-b10754-2459f68
PPL="$PRISM/llama-perplexity"
QUANT=/srv/ml/repos/omnimergekit/tools/opencoti-quantize/bin/opencoti-quantize
META=/srv/ml/scripts/gguf_read_prism_meta.py
FOLD=/srv/ml/scripts/bonsai_hadamard_fold.py
DIR=/mnt/sdc/opencoti-models/sdc-ml/omnimerge-v6
REFD=/mnt/sdc/opencoti-models/bonsai2
SRC=$DIR/Qwen3.8-27B-Omnimerge-v6-BF16.gguf
FOLDED=$DIR/Qwen3.8-27B-Omnimerge-v6-BF16-folded.gguf
TEXT=/srv/ml/ppl-corpus/wiki.test.mid.raw
OUT=/mnt/sdc/opencoti-scratch/v6-ternary/r1
BASE=/mnt/sdc/opencoti-scratch/v6-ternary/bf16-c2048-prism.kld
BASE_PPL=4.8251
mkdir -p "$OUT"
for f in "$PPL" "$QUANT" "$META" "$FOLD" "$SRC" "$TEXT" "$BASE" "$REFD/Ternary-Bonsai-2-27B-PQ2_0.gguf" "$REFD/Ternary-Bonsai-2-27B-PTQ1_0.gguf"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
run_ppl(){ LD_LIBRARY_PATH="$PRISM" CUDA_VISIBLE_DEVICES=0 "$PPL" "$@" -f "$TEXT" -c 2048 --chunks 8 -b 2048 -ub 512 -ngl 99; }
echo "===== V6 FOLD+QUANT $(ts) ====="; df -h /mnt/sdc | tail -1

if [ -s "$FOLDED" ] && grep -q FOLD_OK "$OUT/fold.log" 2>/dev/null; then
  echo "[1 $(ts)] folded GGUF exists and its log says FOLD_OK, reusing"
else
  echo "[1 $(ts)] folding -> $FOLDED"
  rm -f "$FOLDED"
  "$PY" "$FOLD" --src "$SRC" --ref "$REFD/Ternary-Bonsai-2-27B-PQ2_0.gguf" --out "$FOLDED" --device cuda:0 > "$OUT/fold.log" 2>&1
  grep -q FOLD_OK "$OUT/fold.log" || { echo "FATAL fold"; tail -20 "$OUT/fold.log"; exit 1; }
fi
grep -E "^\[contract\]|^\[done\]" "$OUT/fold.log" | sed 's/^/   /'

echo "[G1 $(ts)] LOSSLESS gate — folded bf16 on the prism fork (expect PPL ~ $BASE_PPL)"
run_ppl -m "$FOLDED" > "$OUT/g1-folded-ppl.log" 2>&1 || { echo "FATAL G1 run"; tail -15 "$OUT/g1-folded-ppl.log"; exit 1; }
grep -E "Hadamard" "$OUT/g1-folded-ppl.log" | cut -c1-200 | sed 's/^/   /'
G1=$(grep -oE "Final estimate: PPL = [0-9.]+" "$OUT/g1-folded-ppl.log" | grep -oE "[0-9.]+$")
echo "   folded PPL = ${G1:-NONE}   (bf16 base $BASE_PPL)"
"$PY" -c "import sys; g=float('${G1:-0}'); b=$BASE_PPL; sys.exit(0 if g>0 and abs(g-b)/b<0.01 else 1)" \
  || { echo "FATAL G1: the fold is NOT function-preserving; nothing is quantized from it"; exit 1; }
echo "   G1 PASS"

"$PY" "$META" "$FOLDED" > "$OUT/folded.kv.txt" 2>&1
for T in PQ2_0 PTQ1_0; do
  Q=$DIR/Qwen3.8-27B-Omnimerge-v6-$T-folded.gguf
  REF=$REFD/Ternary-Bonsai-2-27B-$T.gguf
  echo "[2 $(ts)] quantize $T -> $Q"
  /bin/sh "$QUANT" --tensor-type ssm_alpha=bf16 --tensor-type ssm_beta=bf16 \
     --token-embedding-type "$T" --output-tensor-type "$T" \
     "$FOLDED" "$Q" "$T" 32 > "$OUT/quantize-$T.log" 2>&1 || { echo "FATAL quantize $T"; tail -15 "$OUT/quantize-$T.log"; exit 1; }
  sha256sum "$Q" | tee "$Q.sha256" | cut -c1-16 | sed "s/^/   sha /"; echo "   $(stat -c%s "$Q") B  (PrismML $(stat -c%s "$REF") B)"
  "$PY" "$META" "$Q" --tensors > "$OUT/ours-$T.tensors.txt" 2>&1
  "$PY" "$META" "$REF" --tensors > "$OUT/ref-$T.tensors.txt" 2>&1
  head -1 "$OUT/ours-$T.tensors.txt" | sed 's/^/   ours /'; head -1 "$OUT/ref-$T.tensors.txt" | sed 's/^/   ref  /'
  if diff <(tail -n +2 "$OUT/ref-$T.tensors.txt") <(tail -n +2 "$OUT/ours-$T.tensors.txt") > "$OUT/census-$T.diff"; then
    echo "   G2 PASS — per-tensor name/type/shape table IDENTICAL to PrismML's $T"
  else
    echo "FATAL G2: $(grep -c '^>' "$OUT/census-$T.diff") tensors differ"; head -10 "$OUT/census-$T.diff"; exit 1
  fi
  "$PY" "$META" "$Q" > "$OUT/ours-$T.kv.txt" 2>&1; "$PY" "$META" "$REF" > "$OUT/ref-$T.kv.txt" 2>&1
  if diff <(tail -n +2 "$OUT/ref-$T.kv.txt") <(tail -n +2 "$OUT/ours-$T.kv.txt") > "$OUT/kv-$T.diff"; then
    echo "   G3 PASS — prism.hadamard.* identical"
  else
    echo "FATAL G3: prism.hadamard KV differs"; head -10 "$OUT/kv-$T.diff"; exit 1
  fi
  echo "[3 $(ts)] KLD $T (prism fork, fork's own base)"
  run_ppl -m "$Q" --kl-divergence-base "$BASE" --kl-divergence > "$OUT/kld-$T.log" 2>&1 || { echo "   FAIL"; tail -10 "$OUT/kld-$T.log"; continue; }
  grep -E "^(Mean  *PPL\(Q\)|Mean  *KLD|Minimum KLD|Mean  *Δp|RMS Δp|Same top p)" "$OUT/kld-$T.log" | sed 's/^/     /'
done
df -h /mnt/sdc | tail -1
echo "[$(ts)] === V6_FOLD_QUANT_DONE ==="
