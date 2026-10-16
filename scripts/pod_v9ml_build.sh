#!/usr/bin/env bash
# v9-ml quick test BUILDER (POD, CPU only; same recipe as the bs2 run): every arm = the v9 A build recipe with only the drop map changed:
#   expert_drop(map) -> router_shared_upweight alpha 1.2 mlp.down_proj.weight -> official template
#   -> convert f16 -> llama-quantize Q6_K WITHOUT imatrix (user: quick test, no imatrix; A is rebuilt
#   the same way as the baseline arm, so the comparison is no-imatrix vs no-imatrix).
# Arm A must rebuild the published v9 shards byte for byte, or the chain stops.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=""
PY=/workspace/miniconda/envs/omnimergekit/bin/python
OMK=/workspace/omnimergekit
LC=/workspace/llama.cpp
SRC=/workspace/base/gemma-4-26B-A4B-it
V9=/workspace/gemma-4-20B-v9-agentic
TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
Q=/workspace/v9ml_quick
ARMS=(ml_k8 A ml_k3 ml_k8v)
say(){ echo "[build $(date -u +%H:%M:%SZ)] $*"; }
die(){ say "V9ML_BUILD_FAIL $*"; exit 1; }
map_of(){ echo /workspace/v9ml_quick/maps/$1.json; }

[ "$(sha256sum $V9/chat_template.jinja | cut -d' ' -f1)" = "$TSHA" ] || die "v9 template sha"
for arm in "${ARMS[@]}"; do
  D=$Q/$arm; BF=$D/bf16; G=$D/v9ml-$arm-Q6_K.gguf; mkdir -p $D/logs
  [ -f $D/BUILD_OK ] && { say "$arm: BUILD_OK exists, skip"; continue; }
  MAP=$(map_of $arm); [ -f "$MAP" ] || die "$arm map missing $MAP"
  sha256sum "$MAP" > $D/map.sha256; cp "$MAP" $D/drop_map.json
  say "$arm: expert_drop ($(cut -c1-16 $D/map.sha256))"
  rm -rf $BF
  "$PY" $OMK/scripts/expert_drop.py --source-dir $SRC --drop-map "$MAP" --output-dir $BF > $D/logs/expert_drop.log 2>&1 || die "$arm expert_drop"
  "$PY" $OMK/recipes/gemma4/v5_moe_sweep/router_shared_upweight.py --model-dir $BF --alpha 1.2 --target mlp.down_proj.weight \
    > $D/logs/upweight.log 2>&1 || die "$arm upweight"
  rm -f $BF/*.pre_shared_upweight
  ( cd $BF && sha256sum model-*.safetensors > SHA256SUMS ); cp $BF/SHA256SUMS $D/SHA256SUMS; cp $BF/expert_drop_metadata.json $D/ 2>/dev/null
  if [ $arm = A ]; then
    diff <(sort -k2 $BF/SHA256SUMS) <(sort -k2 $V9/SHA256SUMS) > $D/logs/a_vs_published.diff || die "A rebuild differs from the published v9 shards"
    say "A: shards byte-identical to published v9"
  fi
  for f in chat_template.jinja generation_config.json preprocessor_config.json; do cp $V9/$f $BF/$f; done
  [ "$(sha256sum $BF/chat_template.jinja | cut -d' ' -f1)" = "$TSHA" ] || die "$arm template copy"
  say "$arm: convert f16"
  "$PY" $LC/convert_hf_to_gguf.py $BF --outtype f16 --outfile $D/f16.gguf > $D/logs/convert.log 2>&1 || die "$arm convert"
  say "$arm: quantize Q6_K (no imatrix)"
  $LC/build/bin/llama-quantize $D/f16.gguf $G Q6_K > $D/logs/quantize.log 2>&1 || die "$arm quantize"
  grep -qi "imatrix" $D/logs/quantize.log && grep -iE "imatrix" $D/logs/quantize.log | head -3
  sha256sum $G > $G.sha256
  rm -f $D/f16.gguf; rm -rf $BF                      # regenerable from drop_map.json in ~15 min
  touch $D/BUILD_OK; say "$arm: BUILD_OK $(du -h $G | cut -f1)"
done
say "V9ML_BUILD_ALL_OK"
