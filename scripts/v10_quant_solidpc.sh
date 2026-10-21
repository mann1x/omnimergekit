#!/usr/bin/env bash
# v10_quant_solidpc.sh — the v10 pod quant stage (omk scripts/pod_v10_quick_build.sh `quant`) on solidpc's 3090:
# llama.cpp 1268eda (b11707), calibration_datav5, Q4_K_M, the shipped v9-agentic template ae53464b.
# CHUNKS (default 128 = the QUICK v5 imatrix the owner asked for; the pod's -1/822 was bug-818). PROOF=1 runs step 1.
#   1. proof: a_ref from its own bf16 + its own imatrix (6e8ba965) must reproduce the pushed a_ref GGUF (723f9539)
#      tensor for tensor and KV for KV (only quantize.imatrix.file, a path, may differ);
#   2. a_noshared = a_ref without router_shared_upweight: fresh imatrix (all chunks) + Q4_K_M.
set -uo pipefail
CHUNKS=${CHUNKS:-128}; PROOF=${PROOF:-0}
B=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models; V=$B/google/v9_agentic_ab/v10_recipe
PY=/root/anaconda3/envs/omnimergekit/bin/python; OMK=/shared/dev/omnimergekit; LC=/shared/dev/llama.cpp-1268eda
CAL=$OMK/scripts/calibration_datav5.txt; CALSHA=bd925aea03bec5dd; LCREV=1268eda
TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
AREF=$B/results/v10_diag_maps_2026-10-17/gguf/a_ref/gemma-4-20B-v9-agentic-a_ref-Q4_K_M.gguf
say(){ echo "[v10q $(date -u +%FT%TZ)] $*"; }; die(){ say "FATAL $*"; exit 1; }
[ "$(git -C $LC rev-parse --short=7 HEAD)" = $LCREV ] || die "llama.cpp worktree is not $LCREV"
for b in llama-quantize llama-imatrix; do [ -x $LC/build/bin/$b ] || die "missing $LC/build/bin/$b"; done
[ "$(sha256sum $CAL | cut -c1-16)" = $CALSHA ] || die "calib sha"
quant(){ local n=$1; local BF=$V/$n/gemma-4-20B-v9-agentic-$n GG=$V/gguf/$n
  [ "$(sha256sum $BF/chat_template.jinja | cut -d' ' -f1)" = $TSHA ] || die "$n template"
  mkdir -p $GG; say "$n: convert + imatrix (v5, $CHUNKS chunks, $( [ -f $GG/imatrix.dat ] && echo 'REUSED '$(sha256sum $GG/imatrix.dat | cut -c1-16) || echo fresh)) + Q4_K_M on $LCREV"
  CUDA_VISIBLE_DEVICES=0 LLAMA_CPP_HOME=$LC "$PY" $OMK/scripts/quantize_gguf.py --model $BF --output-dir $GG \
    --only Q4_K_M --cal-data $CAL --imatrix-chunks $CHUNKS --imatrix-timeout 28800 --no-kld \
    --chat-template-file $BF/chat_template.jinja --chat-template-sha256 $TSHA \
    --base-model-id google/gemma-4-26B-A4B-it --no-upload --keep-local > $V/$n.quant.log 2>&1 || die "$n quantize_gguf (see $V/$n.quant.log)"
  local Q=$GG/gemma-4-20B-v9-agentic-$n-Q4_K_M.gguf; [ -s $Q ] && [ -s $GG/imatrix.dat ] || die "$n: missing Q4_K_M or imatrix"
  say "$n: QUANT_OK Q4_K_M sha $(sha256sum $Q | cut -c1-16) imatrix sha $(sha256sum $GG/imatrix.dat | cut -c1-16)"; }
if [ "$PROOF" = 1 ]; then
quant a_ref
"$PY" /shared/dev/omnimergekit/scripts/compare_gguf_model.py $AREF $V/gguf/a_ref/gemma-4-20B-v9-agentic-a_ref-Q4_K_M.gguf | tee $V/proof_a_ref.log
grep -q GGUF_SAME_MODEL $V/proof_a_ref.log || die "solidpc toolchain does NOT reproduce a_ref — stop before a_noshared"
say "PROOF_OK: solidpc 1268eda reproduces a_ref"
fi
quant a_noshared
"$PY" -c "
import sys; sys.path.insert(0,'$LC/gguf-py'); from gguf import GGUFReader
r=GGUFReader('$V/gguf/a_noshared/gemma-4-20B-v9-agentic-a_noshared-Q4_K_M.gguf'); f=r.fields
print('a_noshared KV', {k: f[k].contents() for k in ('general.name','general.file_type','quantize.imatrix.dataset','quantize.imatrix.chunks_count') if k in f})"
say "V10Q_DONE"
