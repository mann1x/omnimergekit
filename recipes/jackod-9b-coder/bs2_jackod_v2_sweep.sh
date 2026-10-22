#!/usr/bin/env bash
# JackOD-9B-Coder v2: every GGUF tier -> ManniX-ITA/JackOD-9B-Coder-MTP-GGUF from the v2 full-AC imatrix.
# Q6_K first (it is the tier the card's measurements use), then v1's tier set (bartowski ladder + legacy CD
# ladder incl. the opt-in CD-Q2_K), then imatrix.dat + F16, every remote sha checked. The GGUF README is left
# alone (OMK_NO_README=1): the card is written from the eval results, not generated.
#
#   MODEL=/path/JackOD-9B-Coder OUT=/path/quant_v2 IMATRIX_SHA=<sha256 of OUT/imatrix.dat> LC=/path/llama.cpp-311d4211bf \
#   OMK=/path/omnimergekit bash bs2_jackod_v2_sweep.sh
# The weights dir MUST be named JackOD-9B-Coder: quantize_gguf names the GGUF files and general.name after it.
set -uo pipefail
: "${MODEL:?}" "${OUT:?}" "${IMATRIX_SHA:?}" "${LC:?}" "${OMK:?}"
PY="${PY:-python}"; REPO=ManniX-ITA/JackOD-9B-Coder-MTP-GGUF
# No CD-IQ4_K_M for this model: on an IQ4_XS base it only matches IQ4_XS (0.02608 vs 0.02615,
# 22 MB smaller); on IQ4_NL it gains 0.0004 for +0.23 GB — CD-Q4_K_M covers that slot (bm-884).
TIERS="CD-Q2_K,CD-Q3_K_L,CD-Q4_K_M,CD-Q5_K_M,CD-Q6_K,IQ2_M,IQ2_S,IQ2_XS,IQ3_M,IQ3_XS,IQ3_XXS,IQ4_NL,IQ4_XS,Q2_K_L,Q3_K_L,Q3_K_M,Q3_K_S,Q3_K_XL,Q4_K_L,Q4_K_M,Q4_K_S,Q5_K_L,Q5_K_M,Q5_K_S,Q6_K_L,Q8_0"
[ "$(basename "$MODEL")" = JackOD-9B-Coder ] || { echo "FATAL: weights dir must be named JackOD-9B-Coder"; exit 3; }
[ "$(sha256sum "$OUT/imatrix.dat" | cut -d' ' -f1)" = "$IMATRIX_SHA" ] || { echo "FATAL: staged imatrix is not $IMATRIX_SHA"; exit 3; }
export LLAMA_CPP_HOME="$LC" OMK_NO_README=1
"$PY" "$OMK/scripts/quantize_gguf.py" --model "$MODEL" --repo $REPO --output-dir "$OUT" --only Q6_K --force-imatrix \
    --keep-local --base-model-id Qwen/Qwen3.5-9B || exit 1
"$PY" "$OMK/scripts/quantize_gguf.py" --model "$MODEL" --repo $REPO --output-dir "$OUT" --only "$TIERS" --force-imatrix \
    --base-model-id Qwen/Qwen3.5-9B || exit 1
[ "$(sha256sum "$OUT/imatrix.dat" | cut -d' ' -f1)" = "$IMATRIX_SHA" ] || { echo "FATAL: imatrix changed during the sweep"; exit 3; }
for f in imatrix.dat JackOD-9B-Coder-F16.gguf; do hf upload $REPO "$OUT/$f" "$f" || exit 1; done
echo "JACKOD_V2_SWEEP_DONE"
