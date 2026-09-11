#!/usr/bin/env bash
# JackOD4-9B-Coder Q6_K from the FULL-AC imatrix (02d25d65...).
# imatrix.dat + F16 already sit in the output dir -> pre-stage detection uses the AC one.
set -euo pipefail
LC=/mnt/sdc/llama.cpp-311d4211bf
BF16=/mnt/sdc/oxopus/JackOD4-9B-Coder
OUT=/mnt/sdc/oxopus/jackod4-9b-ac
AC_IMAT_SHA=02d25d657dc8a42f350acebe6d7eff41f6bc9b3e084607644d29c8f3e5958046
OMK=/srv/ml/repos/omnimergekit/scripts/quantize_gguf.py

got=$(sha256sum "$OUT/imatrix.dat" | cut -d" " -f1)
[ "$got" = "$AC_IMAT_SHA" ] || { echo "FATAL: staged imatrix is NOT the AC one: $got"; exit 3; }
echo ">>> AC imatrix verified $got"

export LLAMA_CPP_HOME="$LC"
export CUDA_VISIBLE_DEVICES=0
PY=/root/anaconda3/envs/omnimergekit/bin/python
"$PY" "$OMK" --model "$BF16" --output-dir "$OUT" --only Q6_K --force-imatrix \
    --keep-local --no-upload --base-model-id Qwen/Qwen3.5-9B
echo ">>> artifacts:"; ls -la "$OUT"
echo "JACKOD4_AC_Q6K_DONE $(date -u +%FT%TZ)"
