#!/usr/bin/env bash
# JackOD-9B-Coder v2: full AtomicChat imatrix (an imatrix is a fingerprint of ONE set of weights: v1's 02d25d65...
# does NOT apply to v2). Corpus = AtomicChat/calib-corpora builds/ornith-1.5-9b/calib_train.txt, all 9686 chunks,
# --parse-special (quantize_gguf default), llama.cpp 311d4211bf (cohort pin).
#
#   MODEL=/path/JackOD-9B-Coder OUT=/path/quant_v2 AC=/path/calib_train.txt LC=/path/llama.cpp-311d4211bf \
#   OMK=/path/omnimergekit bash bs2_jackod_v2_ac_imatrix.sh
set -euo pipefail
: "${MODEL:?}" "${OUT:?}" "${AC:?}" "${LC:?}" "${OMK:?}"
AC_SHA=a47fdfeec1a1071ea542f4e77d619a567ab9e6002940262fe273223e105fb0be
PY="${PY:-python}"
[ "$(sha256sum "$AC" | cut -d' ' -f1)" = "$AC_SHA" ] || { echo "FATAL: AC corpus sha mismatch"; exit 3; }
[ "$(git -C "$LC" rev-parse --short=10 HEAD)" = 311d4211bf ] || { echo "FATAL: llama.cpp is not 311d4211bf"; exit 3; }
mkdir -p "$OUT"
# never pre-stage: quantize_gguf.py adopts an existing <out>/imatrix.dat as if it were this run's
[ -e "$OUT/imatrix.dat" ] && { echo "FATAL: imatrix.dat already in $OUT - refusing"; exit 3; }
LLAMA_CPP_HOME="$LC" "$PY" "$OMK/scripts/quantize_gguf.py" --model "$MODEL" --output-dir "$OUT" --imatrix-only \
    --imatrix-chunks -1 --imatrix-timeout 36000 --cal-data "$AC" --keep-local --no-upload --base-model-id Qwen/Qwen3.5-9B
sha256sum "$OUT/imatrix.dat"
echo "JACKOD_V2_AC_IMATRIX_DONE"
