#!/usr/bin/env bash
# JackOD-9B-Coder: full GGUF tier sweep on bs2 -> ManniX-ITA/JackOD-9B-Coder-MTP-GGUF
#
# Q6_K is EXCLUDED: it already exists (built 2026-09-11 from the same F16 with the
# same AC imatrix) and is uploaded separately, so it is not rebuilt here.
#
# The AC imatrix is PRE-STAGED as <output_dir>/imatrix.dat, which quantize_gguf.py
# adopts. That is a trap when the staged file is a partial or a different cut, so it
# is sha-gated against the JackOD4 AC imatrix before anything runs.
set -uo pipefail
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be exported}"

OMK=/srv/ml/repos/omnimergekit/scripts/quantize_gguf.py
[ -f "$OMK" ] || OMK=/mnt/sdc/repos/omnimergekit/scripts/quantize_gguf.py
MODEL=/mnt/sdc/oxopus/JackOD-9B-Coder
OUT=/mnt/sdc/oxopus/jackod_publish
REPO=ManniX-ITA/JackOD-9B-Coder-MTP-GGUF
F16SRC=/mnt/sdc/oxopus/jackod4-9b-ac/JackOD4-9B-Coder-F16.gguf
IMATSRC=/mnt/sdc/oxopus/jackod4-9b-ac/imatrix.dat
AC_IMAT_SHA=02d25d657dc8a42f350acebe6d7eff41f6bc9b3e084607644d29c8f3e5958046
LC=/mnt/sdc/llama.cpp-311d4211bf
L=/mnt/sdc/oxopus/jackod_quant_sweep.log
mkdir -p "$OUT"
exec > >(tee -a "$L") 2>&1
echo ">>> $(date -u +%FT%TZ) JackOD-9B-Coder tier sweep -> $REPO"

[ -f "$OMK" ] || { echo "FATAL: quantize_gguf.py not found"; exit 1; }
[ -d "$MODEL" ] || { echo "FATAL: no weights dir $MODEL"; exit 1; }

# imatrix identity gate BEFORE staging -- an adopted wrong/partial imatrix is silent
got=$(sha256sum "$IMATSRC" | cut -d' ' -f1)
[ "$got" = "$AC_IMAT_SHA" ] || { echo "FATAL: imatrix sha $got != AC $AC_IMAT_SHA"; exit 3; }
echo "    AC imatrix verified $got"

# disk preflight -- a sweep that dies at tier 14 for space wastes hours
FREE=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc '0-9')
echo "    /mnt/sdc free: ${FREE}G"
[ "$FREE" -ge 60 ] || { echo "FATAL: need >=60G free for the sweep (tiers are deleted after upload)"; exit 1; }

[ -e "$OUT/JackOD-9B-Coder-F16.gguf" ] || ln "$F16SRC" "$OUT/JackOD-9B-Coder-F16.gguf"
[ -e "$OUT/imatrix.dat" ] || ln "$IMATSRC" "$OUT/imatrix.dat"
echo "    staged F16 + imatrix into $OUT"

export LLAMA_CPP_HOME="$LC"
export HF_HUB_ENABLE_HF_TRANSFER=1
PY=/root/anaconda3/envs/omnimergekit/bin/python

"$PY" "$OMK" \
    --model "$MODEL" \
    --repo "$REPO" \
    --output-dir "$OUT" \
    --exclude Q6_K \
    --force-imatrix \
    --base-model-id Qwen/Qwen3.5-9B
rc=$?
echo ">>> sweep rc=$rc"
[ $rc -eq 0 ] && echo "JACKOD_SWEEP_DONE $(date -u +%FT%TZ)" || echo "JACKOD_SWEEP_FAILED rc=$rc"
exit $rc
