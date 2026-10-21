#!/usr/bin/env bash
# pod_v10_quick_build.sh — ON THE POD: v10-recipe candidates hub_k12 / hub_k9 -> bf16 -> quick imatrix -> Q4_K_M.
# The models stay gemma-4-20B-v9-agentic (owner: "it's still a v9-agentic, don't change to v10, that's only the recipe"):
# gemma-4-20B-v9-agentic-<map>.
# Owner 2026-10-17: "build also the 2nd map without multi-lingual / make a quick i-matrix with v5 calib, create
# for both also the q4_k_m quant" + "make sure the quantize is from the latest llama.cpp not b9700".
#   bf16  = expert_drop <map> -> router_shared_upweight a1.2 (the v9-agentic recipe, only the map changes),
#           + the shipped v9-agentic chat_template / generation_config / preprocessor_config (official template)
#   quant = quantize_gguf.py with LLAMA_CPP_HOME=llama.cpp-latest (b11707): convert -> F16 -> imatrix on
#           calibration_datav5.txt, IMATRIX_CHUNKS chunks (default 128 = the QUICK imatrix the owner asked for) on
#           GPU0+GPU1 -> Q4_K_M with that imatrix. --no-kld (Gemma-4 A4B).
#           bug-818: the 2026-10-17/18 pod builds (a_ref, a_ml3, a_core5, hub_k9, hub_k12) ran --imatrix-chunks -1
#           (all 822 chunks). Only the AC imatrix goes full corpus; IMATRIX_CHUNKS=-1 reproduces those pushed builds.
#           imatrix.dat is kept next to the quant; the F16 intermediate is removed after the Q4_K_M is verified.
# usage: pod_v10_quick_build.sh base | bf16 <name> | quant <name> | rmbase
set -uo pipefail
W=/workspace; PY=$W/miniconda/envs/omnimergekit/bin/python; OMK=$W/omnimergekit; LC=$W/llama.cpp-latest
V=$W/v10; BASE=$V/base/gemma-4-26B-A4B-it; BREV=4d7ae4984b7db7de8f8457170b3f1a419ee76d52
V9=$W/gemma-4-20B-v9-agentic; TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
CAL=$OMK/scripts/calibration_datav5.txt; CALSHA=bd925aea03bec5dd; LCREV=1268eda
declare -A MSHA=([hub_k12]=9190847f59683df0 [hub_k9]=5ee3299374a74e86 [a_core5]=4044c1562814015f [a_ml3]=0578361d58affa3c [a_ref]=24c7b6cb8c707152)
say(){ echo "[v10 $(date -u +%FT%TZ)] $*"; }
die(){ say "V10_FAIL $*"; exit 1; }
mkdir -p $V/logs
case ${1:?stage} in
base)
  "$PY" $OMK/scripts/pod_hf_snapshot_fetch.py google/gemma-4-26B-A4B-it $BREV $BASE > $V/logs/base_fetch.log 2>&1 \
    || die "base fetch (see $V/logs/base_fetch.log)"
  [ -f $BASE/FETCH_OK ] || die "no FETCH_OK"; say "BASE_OK $(tail -n1 $V/logs/base_fetch.log)";;
bf16)
  n=${2:?name}; M=$V/maps/$n.json; NAME=gemma-4-20B-v9-agentic-$n; BF=$V/$n/$NAME
  [ -n "${MSHA[$n]:-}" ] || die "unknown map $n"
  [ "$(sha256sum $M | cut -c1-16)" = "${MSHA[$n]}" ] || die "$n map sha"
  [ -f $BASE/FETCH_OK ] || die "base not fetched"
  [ -f $BF/BUILD_OK ] && { say "$n: bf16 BUILD_OK exists, skip"; exit 0; }
  rm -rf $BF; mkdir -p $V/$n
  say "$n: expert_drop (${MSHA[$n]})"
  CUDA_VISIBLE_DEVICES="" "$PY" $OMK/scripts/expert_drop.py --source-dir $BASE --drop-map $M --output-dir $BF \
    > $V/logs/$n.expert_drop.log 2>&1 || die "$n expert_drop"
  CUDA_VISIBLE_DEVICES="" "$PY" $OMK/recipes/gemma4/v5_moe_sweep/router_shared_upweight.py --model-dir $BF --alpha 1.2 \
    --target mlp.down_proj.weight > $V/logs/$n.upweight.log 2>&1 || die "$n upweight"
  rm -f $BF/*.pre_shared_upweight
  for f in chat_template.jinja generation_config.json preprocessor_config.json; do cp $V9/$f $BF/$f; done
  [ "$(sha256sum $BF/chat_template.jinja | cut -d' ' -f1)" = $TSHA ] || die "$n template copy"
  ( cd $BF && sha256sum model-*.safetensors > SHA256SUMS )
  touch $BF/BUILD_OK; say "$n: BF16_OK $(du -sh $BF | cut -f1)";;
quant)
  n=${2:?name}; NAME=gemma-4-20B-v9-agentic-$n; BF=$V/$n/$NAME; GG=$V/gguf/$n; Q=$GG/$NAME-Q4_K_M.gguf
  [ -f $BF/BUILD_OK ] || die "$n: no bf16"
  [ "$(git -C $LC rev-parse --short=7 HEAD)" = $LCREV ] || die "llama.cpp-latest is not $LCREV"
  for b in llama-quantize llama-imatrix llama-server; do [ -x $LC/build/bin/$b ] || die "missing $LC/build/bin/$b"; done
  [ "$(sha256sum $CAL | cut -c1-16)" = $CALSHA ] || die "calib sha"
  [ -f $GG/imatrix.dat ] && die "$GG/imatrix.dat exists — refusing to clobber an imatrix"
  mkdir -p $GG
  say "$n: convert + imatrix (v5, ${IMATRIX_CHUNKS:-128} chunks, GPU0+GPU1) + Q4_K_M on llama.cpp b11707 ($LCREV)"
  CUDA_VISIBLE_DEVICES=0,1 LLAMA_CPP_HOME=$LC "$PY" $OMK/scripts/quantize_gguf.py --model $BF --output-dir $GG \
    --only Q4_K_M --cal-data $CAL --imatrix-chunks ${IMATRIX_CHUNKS:-128} --imatrix-timeout 14400 --no-kld \
    --chat-template-file $BF/chat_template.jinja --chat-template-sha256 $TSHA \
    --base-model-id google/gemma-4-26B-A4B-it --no-upload --keep-local > $V/logs/$n.quant.log 2>&1 \
    || die "$n quantize_gguf (see $V/logs/$n.quant.log)"
  [ -s $GG/imatrix.dat ] || die "$n: no imatrix.dat"
  [ -s $Q ] || die "$n: no Q4_K_M"
  kv=$(PYTHONPATH=$LC/gguf-py "$PY" -c "
from gguf import GGUFReader
r=GGUFReader('$Q'); f=r.fields
g=lambda k: f[k].contents() if k in f else None
print(g('general.file_type'), g('quantize.imatrix.file'), g('quantize.imatrix.chunks_count'))")
  say "$n: Q4_K_M KV file_type/imatrix.file/chunks = $kv"
  grep -q "imatrix.dat" <<<"$kv" || die "$n: Q4_K_M carries no quantize.imatrix.file"
  rm -f $GG/$NAME-F16.gguf
  say "$n: QUANT_OK $(du -h $Q | cut -f1) Q4_K_M sha $(sha256sum $Q | cut -c1-16) imatrix sha $(sha256sum $GG/imatrix.dat | cut -c1-16)";;
rmbase)
  rm -rf $V/base; say "base removed (re-fetchable: google/gemma-4-26B-A4B-it@${BREV:0:10})";;
*) die "unknown stage $1";;
esac
