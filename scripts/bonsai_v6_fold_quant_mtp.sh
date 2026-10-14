#!/usr/bin/env bash
# bonsai_v6_fold_quant_mtp.sh — the MTP variant of bonsai_v6_fold_quant.sh.
#
# v6 ships its own MTP (nextn) head: 15 tensors at blk.64. Owner's rule: THE HEAD IS LEFT AS IS.
# So the head is grafted from v6's own `mtp.*` weights (convert_hf_to_gguf --mtp, bf16) and is
#   - NOT folded      (it is not in prism.hadamard.weight_names; the fold tool appends it verbatim)
#   - NOT ternarized  (--tensor-type pins blk.64 to bf16; its norms stay F32)
# and that is PROVEN, not requested: gate G4 hashes every blk.64.* tensor's bytes in each final
# file against the head GGUF and requires 15/15 identical.
# The body is identical in construction to the non-MTP run; G2 checks its 851 rows against
# PrismML's table, the 15 head rows being the only permitted additions.
set -uo pipefail
ts(){ date -u '+%H:%M:%SZ'; }
PY=/root/anaconda3/envs/omnimergekit/bin/python
export PYTHONPATH=/srv/ml/dca-cleanroom/llama.cpp/gguf-py
PRISM=/srv/ml/bonsai-assess/ref-b10754/llama-prism-b10754-2459f68
PPL="$PRISM/llama-perplexity"
QUANT=/srv/ml/repos/omnimergekit/tools/opencoti-quantize/bin/opencoti-quantize
META=/srv/ml/scripts/gguf_read_prism_meta.py
SHA=/srv/ml/scripts/gguf_tensor_sha.py
FOLD=/srv/ml/scripts/bonsai_hadamard_fold.py
DIR=/mnt/sdc/opencoti-models/sdc-ml/omnimerge-v6
REFD=/mnt/sdc/opencoti-models/bonsai2
SRC=$DIR/Qwen3.8-27B-Omnimerge-v6-BF16.gguf
HEAD=$DIR/mtp-Omnimerge-v6-BF16.gguf
FOLDED=$DIR/Qwen3.8-27B-Omnimerge-v6-BF16-folded-MTP.gguf
TEXT=/srv/ml/ppl-corpus/wiki.test.mid.raw
OUT=/mnt/sdc/opencoti-scratch/v6-ternary/r1
BASE=/mnt/sdc/opencoti-scratch/v6-ternary/bf16-c2048-prism.kld
BASE_PPL=4.8251
for f in "$PPL" "$QUANT" "$META" "$SHA" "$FOLD" "$SRC" "$HEAD" "$TEXT" "$BASE"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
run_ppl(){ LD_LIBRARY_PATH="$PRISM" CUDA_VISIBLE_DEVICES=0 "$PPL" "$@" -f "$TEXT" -c 2048 --chunks 8 -b 2048 -ub 512 -ngl 99; }
echo "===== V6 FOLD+QUANT (MTP) $(ts) ====="; df -h /mnt/sdc | tail -1
"$PY" "$SHA" "$HEAD" blk.64. > "$OUT/head.sha.txt" || { echo "FATAL head sha"; exit 1; }
echo "   head: $(wc -l < "$OUT/head.sha.txt") tensors hashed from $HEAD"

echo "[1 $(ts)] fold + graft -> $FOLDED"
rm -f "$FOLDED"
"$PY" "$FOLD" --src "$SRC" --ref "$REFD/Ternary-Bonsai-2-27B-PQ2_0.gguf" --mtp-head "$HEAD" --out "$FOLDED" --device cuda:0 > "$OUT/fold-mtp.log" 2>&1
grep -q FOLD_OK "$OUT/fold-mtp.log" || { echo "FATAL fold"; tail -20 "$OUT/fold-mtp.log"; exit 1; }
grep -E "^\[(contract|mtp|done)\]" "$OUT/fold-mtp.log" | sed 's/^/   /'

echo "[G1 $(ts)] LOSSLESS gate (expect PPL ~ $BASE_PPL)"
run_ppl -m "$FOLDED" > "$OUT/g1-folded-mtp-ppl.log" 2>&1 || { echo "FATAL G1 run"; tail -15 "$OUT/g1-folded-mtp-ppl.log"; exit 1; }
G1=$(grep -oE "Final estimate: PPL = [0-9.]+" "$OUT/g1-folded-mtp-ppl.log" | grep -oE "[0-9.]+$")
echo "   folded+MTP PPL = ${G1:-NONE}"
"$PY" -c "import sys; g=float('${G1:-0}'); b=$BASE_PPL; sys.exit(0 if g>0 and abs(g-b)/b<0.01 else 1)" || { echo "FATAL G1"; exit 1; }
echo "   G1 PASS"

for T in PQ2_0 PTQ1_0; do
  Q=$DIR/Qwen3.8-27B-Omnimerge-v6-$T-folded-MTP.gguf
  REF=$REFD/Ternary-Bonsai-2-27B-$T.gguf
  echo "[2 $(ts)] quantize $T -> $Q"
  /bin/sh "$QUANT" --tensor-type ssm_alpha=bf16 --tensor-type ssm_beta=bf16 --tensor-type 'blk\.64\.=bf16' \
     --token-embedding-type "$T" --output-tensor-type "$T" \
     "$FOLDED" "$Q" "$T" 32 > "$OUT/quantize-$T-mtp.log" 2>&1 || { echo "FATAL quantize $T"; tail -15 "$OUT/quantize-$T-mtp.log"; exit 1; }
  sha256sum "$Q" | tee "$Q.sha256" | cut -c1-16 | sed "s/^/   sha /"; echo "   $(stat -c%s "$Q") B"
  "$PY" "$META" "$Q" --tensors > "$OUT/ours-$T-mtp.tensors.txt" 2>&1
  "$PY" "$META" "$REF" --tensors > "$OUT/ref-$T.tensors.txt" 2>&1
  head -1 "$OUT/ours-$T-mtp.tensors.txt" | sed 's/^/   ours /'
  if diff <(tail -n +2 "$OUT/ref-$T.tensors.txt") <(tail -n +2 "$OUT/ours-$T-mtp.tensors.txt" | grep -v '^  blk\.64\.') > "$OUT/census-$T-mtp.diff"; then
    echo "   G2 PASS — 851 body rows identical to PrismML's $T; head rows: $(grep -c '^  blk\.64\.' "$OUT/ours-$T-mtp.tensors.txt")"
  else
    echo "FATAL G2"; head -10 "$OUT/census-$T-mtp.diff"; exit 1
  fi
  "$PY" "$SHA" "$Q" blk.64. > "$OUT/ours-$T-mtp.head.sha.txt"
  if diff "$OUT/head.sha.txt" "$OUT/ours-$T-mtp.head.sha.txt" > "$OUT/head-$T.diff"; then
    echo "   G4 PASS — MTP head LEFT AS IS: $(wc -l < "$OUT/head.sha.txt")/$(wc -l < "$OUT/head.sha.txt") tensors byte-identical to v6's head (type, shape, sha)"
  else
    echo "FATAL G4: the head was altered"; head -10 "$OUT/head-$T.diff"; exit 1
  fi
  echo "[3 $(ts)] KLD $T-MTP"
  run_ppl -m "$Q" --kl-divergence-base "$BASE" --kl-divergence > "$OUT/kld-$T-mtp.log" 2>&1 || { echo "   FAIL"; tail -10 "$OUT/kld-$T-mtp.log"; continue; }
  grep -E "^(Mean  *PPL\(Q\)|Mean  *KLD|Minimum KLD|Mean  *Δp|RMS Δp|Same top p)" "$OUT/kld-$T-mtp.log" | sed 's/^/     /'
done
df -h /mnt/sdc | tail -1
echo "[$(ts)] === V6_FOLD_QUANT_MTP_DONE ==="
