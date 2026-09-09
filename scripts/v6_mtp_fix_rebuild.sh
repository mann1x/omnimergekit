#!/usr/bin/env bash
# Re-quantize the 5 v6 tiers whose MTP head lost its Q4_K floor.
#
# ROOT CAUSE (2026-09-09): the AC rebuild passed --model as an HF REPO ID, so
# detect_mtp's local-dir fallback never fired and mtp_info stayed None. That dropped
# the `--tensor-type blk.64.=q4_K` override:
#   IQ3_XS IQ3_XXS IQ2_M IQ2_S -> llama-quantize BAILED (missing imatrix on blk.64)
#   Q2_K                       -> did NOT bail; built and OVERWROTE a good HF file
#                                 with an MTP head at Q2_K instead of Q4_K.
# The engine now falls back to reading the fact from the F16 GGUF's own KV.
set -euo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
ENGINE=/srv/ml/repos/omnimergekit/scripts/quantize_gguf.py
W=/mnt/sdc/ml/omnimerge-v6
F16=$W/Qwen3.8-27B-Omnimerge-v6-F16.gguf
TIERS="Q2_K,IQ3_XS,IQ3_XXS,IQ2_M,IQ2_S"

"$PY" /srv/ml/scripts/v6_mtp_gate.py "$F16"

IMAT_SHA=$(sha256sum "$W/imatrix.dat" | cut -d" " -f1)
echo "[gate] imatrix sha256=${IMAT_SHA:0:16}"

# llama-quantize bailed at blk.64 -- the LAST block -- so those outputs are truncated
# yet carry a valid magic: exactly the shape mistaken for a finished build.
for t in IQ3_XS IQ3_XXS IQ2_M IQ2_S; do
  f="$W/Qwen3.8-27B-Omnimerge-v6-$t.gguf"
  if [ -f "$f" ]; then
    echo "[clean] removing partial $(basename "$f") ($(stat -c%s "$f") B)"
    rm -f "$f"
  fi
done

echo "[run] tiers: $TIERS"
"$PY" "$ENGINE" --model ManniX-ITA/Qwen3.8-27B-Omnimerge-v6 \
  --repo ManniX-ITA/Qwen3.8-27B-Omnimerge-v6-MTP-GGUF \
  --output-dir "$W" --only "$TIERS" \
  --cal-data "$W/calib_train.txt" --imatrix-chunks -1 \
  --no-sanity-check --threads 32 --ngl 99
echo "V6_MTP_FIX_DONE"
