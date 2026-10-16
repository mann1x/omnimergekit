#!/usr/bin/env bash
# v9-ml ml_k8: rebuild the BF16 model dir and compute its AC imatrix (POD, BOTH GPUs: owner 2026-10-15).
# Build = v9ml_pod_build.sh for arm ml_k8 (expert_drop -> router_shared_upweight a1.2 -> official
# template). Gate: the rebuilt shards must equal the ml_k8 build that was evaluated
# (/workspace/v9ml_quick/ml_k8/SHA256SUMS). imatrix = pod_v9_agentic_publish.sh's imatrix stage
# verbatim (AtomicChat calib_train.txt, all chunks, --parse-special, save every 1000) — only the
# model and output dir change. Both 3090s, as v9 A's imatrix ran (whole F16 on GPU).
set -uo pipefail
W=/workspace; PY=$W/miniconda/envs/omnimergekit/bin/python; OMK=$W/omnimergekit; LC=$W/llama.cpp
SRC=$W/base/gemma-4-26B-A4B-it; V9=$W/gemma-4-20B-v9-agentic
NAME=gemma-4-20B-v9ml-ml_k8; BF=$W/v9ml_k8/$NAME; GG=$W/gguf/$NAME
MAP=$W/v9ml_quick/maps/ml_k8.json; MSHA=84f2bd961772adbd
REF=$W/v9ml_quick/ml_k8/SHA256SUMS
TMPL=$W/stage/tmpl/chat_template.jinja
TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
CALIB=/shared/dev/calib-corpora/builds/gemma-4-20b-v9-agentic/calib_train.txt; CSHA=33fef70c1f8d5965
say(){ echo "[k8imat $(date -u +%H:%M:%SZ)] $*"; }
die(){ say "K8IMAT_FAIL $*"; exit 1; }
[ "$(sha256sum $MAP | cut -c1-16)" = $MSHA ] || die "map sha"
[ "$(sha256sum $TMPL | cut -d' ' -f1)" = $TSHA ] || die "template sha"
[ "$(sha256sum $CALIB | cut -c1-16)" = $CSHA ] || die "calib sha"
[ -f "$GG/imatrix.dat" ] && die "$GG/imatrix.dat exists — refusing to clobber an imatrix"
mkdir -p $W/v9ml_k8/logs "$GG"
if [ ! -f $BF/BUILD_OK ]; then
  say "expert_drop ml_k8 ($MSHA)"; rm -rf $BF
  CUDA_VISIBLE_DEVICES="" "$PY" $OMK/scripts/expert_drop.py --source-dir $SRC --drop-map $MAP --output-dir $BF \
    > $W/v9ml_k8/logs/expert_drop.log 2>&1 || die "expert_drop"
  CUDA_VISIBLE_DEVICES="" "$PY" $OMK/recipes/gemma4/v5_moe_sweep/router_shared_upweight.py --model-dir $BF --alpha 1.2 \
    --target mlp.down_proj.weight > $W/v9ml_k8/logs/upweight.log 2>&1 || die "upweight"
  rm -f $BF/*.pre_shared_upweight
  ( cd $BF && sha256sum model-*.safetensors > SHA256SUMS )
  diff <(sort -k2 $BF/SHA256SUMS) <(sort -k2 $REF) > $W/v9ml_k8/logs/shards_vs_evaluated.diff || die "rebuilt shards differ from the evaluated ml_k8"
  say "shards byte-identical to the evaluated ml_k8"
  for f in chat_template.jinja generation_config.json preprocessor_config.json; do cp $V9/$f $BF/$f; done
  [ "$(sha256sum $BF/chat_template.jinja | cut -d' ' -f1)" = $TSHA ] || die "template copy"
  touch $BF/BUILD_OK
fi
say "imatrix (AtomicChat, full corpus, --parse-special) on GPU0+GPU1"
CUDA_VISIBLE_DEVICES=0,1 LLAMA_CPP_HOME=$LC "$PY" $OMK/scripts/quantize_gguf.py \
  --model "$BF" --output-dir "$GG" --imatrix-only \
  --cal-data "$CALIB" --imatrix-chunks -1 --imatrix-save-frequency 1000 --imatrix-timeout 172800 \
  --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
  --base-model-id google/gemma-4-26B-A4B-it --no-upload --keep-local \
  > $W/logs/imatrix_ml_k8.out 2>&1 || die "imatrix (see logs/imatrix_ml_k8.out)"
grep -q IMATRIX_ONLY_DONE $W/logs/imatrix_ml_k8.out || die "no IMATRIX_ONLY_DONE marker"
ls -la "$GG"; sha256sum "$GG"/imatrix.dat
say "K8IMAT_OK"
