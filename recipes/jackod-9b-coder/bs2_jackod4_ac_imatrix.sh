#!/usr/bin/env bash
# bs2 GPU1 — full-AC imatrix for JackOD4-9B-Coder (NOT JackOD3.5).
# Target basis: match jackod-9b, the benched toolbench arm (calib_train, 9686 chunks).
set -euo pipefail

LC=/mnt/sdc/llama.cpp-311d4211bf
BF16=/mnt/sdc/oxopus/JackOD4-9B-Coder
F16SRC=/mnt/sdc/oxopus/jackod4-9b/JackOD4-9B-Coder-F16.gguf
AC=/mnt/sdc/ml/ornith/calib_train.txt
OUT=/mnt/sdc/oxopus/jackod4-9b-ac
AC_SHA=a47fdfeec1a1071ea542f4e77d619a567ab9e6002940262fe273223e105fb0be
OMK=/srv/ml/repos/omnimergekit/scripts/quantize_gguf.py

echo ">>> $(date -u +%FT%TZ) TARGET MODEL: $BF16   (JackOD4)"
got=$(sha256sum "$AC" | cut -d" " -f1)
[ "$got" = "$AC_SHA" ] || { echo "FATAL: AC corpus sha mismatch: $got"; exit 3; }
echo "    AC corpus OK $got"

"$LC/build/bin/llama-imatrix" --version 2>&1 | grep -q "311d4211b" || { echo "FATAL: wrong llama.cpp"; exit 3; }
echo "    llama.cpp cohort pin OK"

mkdir -p "$OUT"
# Reuse the already-built F16 (hardlink, same fs, zero copy, no re-convert).
# Deliberately do NOT stage any imatrix.dat here: quantize_gguf.py pre-stage
# detection would silently adopt the calib_v5 one as if it were the AC run.
[ -e "$OUT/JackOD4-9B-Coder-F16.gguf" ] || ln "$F16SRC" "$OUT/JackOD4-9B-Coder-F16.gguf"
[ -e "$OUT/imatrix.dat" ] && { echo "FATAL: imatrix.dat already in $OUT - refusing to pre-stage"; exit 3; }
ls -la "$OUT/"

export LLAMA_CPP_HOME="$LC"
export CUDA_VISIBLE_DEVICES=1
PY=/root/anaconda3/envs/omnimergekit/bin/python

echo ">>> $(date -u +%FT%TZ) AC imatrix, chunks=-1 (full 9686)"
"$PY" "$OMK" \
    --model "$BF16" \
    --output-dir "$OUT" \
    --imatrix-only \
    --imatrix-chunks -1 \
    --imatrix-timeout 36000 \
    --cal-data "$AC" \
    --keep-local \
    --no-upload \
    --base-model-id Qwen/Qwen3.5-9B

echo "JACKOD4_AC_BS2_DONE $(date -u +%FT%TZ)"
