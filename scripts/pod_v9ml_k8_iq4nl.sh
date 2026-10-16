#!/usr/bin/env bash
# v9-ml ml_k8 IQ4_NL: waits for the suite's Q6_K to pass its gates ("Q6_K OK" in suite.log), then
# quantizes IQ4_NL from the SAME F16 with the SAME AC imatrix.dat (pre-staged in $GG). CPU ONLY
# (CUDA_VISIBLE_DEVICES="") so neither llama-quantize nor the sanity server takes VRAM from the evals.
# Owner 2026-10-15: "after the q6_k, build also the iq4_nl and copy it to solidpc in .opencoti\models".
set -uo pipefail
W=/workspace; S=$W/v9ml_k8_suite; PY=$W/miniconda/envs/omnimergekit/bin/python; OMK=$W/omnimergekit
NAME=gemma-4-20B-v9ml-ml_k8; BF=$W/v9ml_k8/$NAME; GG=$W/gguf/$NAME; Q=$GG/$NAME-IQ4_NL.gguf
TMPL=$W/stage/tmpl/chat_template.jinja; TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
L=$S/logs/iq4nl.log
say(){ echo "[iq4nl $(date -u +%FT%TZ)] $*" | tee -a $L; }
die(){ say "IQ4NL_FAIL $*"; exit 1; }
mkdir -p $S/logs
say "waiting for the suite's 'Q6_K OK'"
until grep -q "Q6_K OK" $S/logs/suite.log 2>/dev/null; do
  grep -q SUITE_FAIL $S/logs/suite.log 2>/dev/null && die "suite failed before Q6_K OK"
  tmux has-session -t v9k8s 2>/dev/null || grep -q "Q6_K OK" $S/logs/suite.log || die "suite session gone without Q6_K OK"
  sleep 60
done
[ -s $GG/imatrix.dat ] && [ -s $GG/$NAME-F16.gguf ] || die "missing imatrix.dat or F16"
[ -e "$Q" ] && die "$Q already exists (no clobber)"
say "imatrix sha $(sha256sum $GG/imatrix.dat | cut -c1-16); quantize IQ4_NL (CPU only, 16 threads)"
CUDA_VISIBLE_DEVICES="" LLAMA_CPP_HOME=$W/llama.cpp "$PY" $OMK/scripts/quantize_gguf.py \
  --model "$BF" --output-dir "$GG" --only IQ4_NL --no-upload --keep-local --threads 16 \
  --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
  --base-model-id google/gemma-4-26B-A4B-it > $S/logs/quant_iq4nl.out 2>&1 || die "quantize (see logs/quant_iq4nl.out)"
[ -s "$Q" ] || die "no $Q"
grep -qE "^\s+IQ4_NL: sanity PASS" $S/logs/quant_iq4nl.out || die "IQ4_NL sanity not PASS (see logs/quant_iq4nl.out)"
"$PY" - "$Q" <<'PY' | tee -a $L || die "IQ4_NL GGUF does not record the imatrix"
import sys
from gguf import GGUFReader
f = GGUFReader(sys.argv[1]).fields
def s(k):
    v = f.get(k)
    return None if v is None else bytes(v.parts[v.data[0]]).decode("utf-8", "replace")
im, ft = s("quantize.imatrix.file"), f.get("general.file_type")
print("quantize.imatrix.file =", im, "| general.file_type =", None if ft is None else int(ft.parts[ft.data[0]][0]))
sys.exit(0 if im and im.endswith("imatrix.dat") else 1)
PY
(cd $GG && sha256sum $NAME-IQ4_NL.gguf > $NAME-IQ4_NL.gguf.sha256)
say "IQ4NL_OK $(stat -c %s $Q) bytes sha $(cut -c1-16 $Q.sha256)"
