#!/usr/bin/env bash
# Upload the ALREADY-BUILT Q6_K + the AC imatrix to the GGUF repo.
# The tier sweep excludes Q6_K precisely so this artifact is not rebuilt:
# it was quantized from the same F16 with the same AC imatrix, and it is the
# file every tool-calling cell in the model card was measured on.
#
# NAMING: the published filename follows the REPO name, never the build-time
# arm codename -- JackOD-9B-Coder-Q6_K.gguf, not JackOD4-9B-Coder-Q6_K.gguf.
set -uo pipefail
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be exported}"
REPO=ManniX-ITA/JackOD-9B-Coder-MTP-GGUF
AC=/mnt/sdc/oxopus/jackod4-9b-ac
STAGE=/mnt/sdc/oxopus/jackod_q6k_upload
WANT_SHA=abd45a375a1e19193a586bde088c1960f51935f9e7acb0a83e9c3321a031b248
AC_IMAT_SHA=02d25d657dc8a42f350acebe6d7eff41f6bc9b3e084607644d29c8f3e5958046
L=/mnt/sdc/oxopus/upload_jackod_q6k.log
mkdir -p "$STAGE"
exec > >(tee -a "$L") 2>&1
echo ">>> $(date -u +%FT%TZ) Q6_K + imatrix -> $REPO"

got=$(sha256sum "$AC/JackOD4-9B-Coder-Q6_K.gguf" | cut -d' ' -f1)
[ "$got" = "$WANT_SHA" ] || { echo "FATAL: Q6_K sha $got != $WANT_SHA"; exit 3; }
igot=$(sha256sum "$AC/imatrix.dat" | cut -d' ' -f1)
[ "$igot" = "$AC_IMAT_SHA" ] || { echo "FATAL: imatrix sha $igot != AC"; exit 3; }
echo "    Q6_K sha OK / AC imatrix sha OK"

[ -e "$STAGE/JackOD-9B-Coder-Q6_K.gguf" ] || ln "$AC/JackOD4-9B-Coder-Q6_K.gguf" "$STAGE/JackOD-9B-Coder-Q6_K.gguf"
[ -e "$STAGE/imatrix.dat" ] || ln "$AC/imatrix.dat" "$STAGE/imatrix.dat"
printf '%s  %s\n' "$WANT_SHA" "JackOD-9B-Coder-Q6_K.gguf" > "$STAGE/JackOD-9B-Coder-Q6_K.gguf.sha256"
cat "$STAGE/JackOD-9B-Coder-Q6_K.gguf.sha256"

export HF_HUB_ENABLE_HF_TRANSFER=1
rc=0
hf upload "$REPO" "$STAGE/JackOD-9B-Coder-Q6_K.gguf"        JackOD-9B-Coder-Q6_K.gguf        --repo-type model || rc=1
hf upload "$REPO" "$STAGE/JackOD-9B-Coder-Q6_K.gguf.sha256" JackOD-9B-Coder-Q6_K.gguf.sha256 --repo-type model || rc=1
hf upload "$REPO" "$STAGE/imatrix.dat"                      imatrix.dat                      --repo-type model || rc=1
[ $rc -eq 0 ] && echo "JACKOD_Q6K_UPLOAD_DONE $(date -u +%FT%TZ)" || echo "JACKOD_Q6K_UPLOAD_FAILED rc=$rc"
exit $rc
