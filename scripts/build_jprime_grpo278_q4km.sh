#!/usr/bin/env bash
# Merge the GRPO-efficiency LoRA (checkpoint-278) into Jprime-p3-bf16 and build a
# Q4_K_M GGUF for hands-on cline testing.
#
# Runs ON bs2. Two documented traps are handled explicitly:
#   * LoRA round-trip: apply_lora_to_safetensors.py adds scale*(B@A) IN PLACE and writes
#     every other byte through, so the tensor inventory cannot change (no transformers
#     save_pretrained, hence no dropped/triple-nested prefixes). Base has 0 mtp.*/visual.*.
#   * EOG/EOS: quantize_gguf.py normalises Gemma-4 to eos=106 / eot=1 by default.
#     --no-eog-normalise is NOT passed. The KV is verified after the build.
# imatrix: this arm has DIFFERENT weights from the base, so it gets its OWN imatrix.
# Reusing the base's would be invalid. imatrix.dat is kept next to the quant (mandatory).
set -euo pipefail

REPO=/srv/ml/repos/omnimergekit
PY_MERGE=/srv/ml/envs/envs/omk-grpo/bin/python3.11
PY_QUANT=/srv/ml/envs/envs/omnimergekit/bin/python

BASE=/mnt/sdc/v7rework/arms/Jprime-p3-bf16
CKPT=/mnt/sdc/v7rework/grpo_full_lr1e-5/checkpoint-278
NAME=Jprime-grpo-eff-278
MERGED=/mnt/sdc/v7rework/arms/${NAME}-bf16
OUT=/mnt/sdc/v7rework/quants/${NAME}-bf16

for d in "$BASE" "$CKPT"; do [ -d "$d" ] || { echo "FATAL: missing $d"; exit 1; }; done

AVAIL=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc '0-9')
[ "$AVAIL" -ge 120 ] || { echo "FATAL: only ${AVAIL}G free on /mnt/sdc, need >=120G"; exit 1; }
echo ">>> preflight ok: ${AVAIL}G free"

# ---------------------------------------------------------------- 1. merge
if [ -f "$MERGED/.MERGE_DONE" ]; then
  echo ">>> merge already done, skipping"
else
  echo ">>> $(date -u +%FT%TZ) merging LoRA $CKPT -> $MERGED"
  rm -rf "$MERGED"
  "$PY_MERGE" "$REPO/scripts/apply_lora_to_safetensors.py" "$BASE" "$CKPT" "$MERGED"
  # The base dir declares eos_token=<eos> (id 1); the checkpoint declares <turn|> (id 106),
  # which is Gemma-4's real eot. Carry the checkpoint's config so the HF dir is
  # self-consistent. The GGUF EOG is set by the engine's normaliser regardless.
  cp -f "$CKPT/tokenizer_config.json" "$MERGED/tokenizer_config.json"
  touch "$MERGED/.MERGE_DONE"
fi

echo ">>> merged tensor census"
"$PY_MERGE" - "$MERGED" <<'PY'
import json,glob,sys,collections
d=sys.argv[1]; w=json.load(open(glob.glob(d+"/model*.index.json")[0]))["weight_map"]
print("   tensors=%d  mtp.*=%d  visual.*=%d"%(
    len(w),sum(1 for k in w if k.startswith("mtp.")),sum(1 for k in w if k.startswith("visual."))))
PY

# ---------------------------------------------------------------- 2. quantize
echo ">>> $(date -u +%FT%TZ) quantize Q4_K_M (own imatrix, EOG normalise ON)"
cd "$REPO"
export OMK_NO_README=1   # --no-upload: README/repo creation is skipped anyway
"$PY_QUANT" scripts/quantize_gguf.py \
    --model "$MERGED" \
    --base-model-id ManniX-ITA/gemma-4-A4B-98e-v7-coderx-it \
    --output-dir "$OUT" \
    --only Q4_K_M \
    --no-upload \
    --keep-local \
    --cal-data "$REPO/scripts/calibration_datav5.txt" \
    --ngl 99

echo ">>> $(date -u +%FT%TZ) BUILD DONE"
ls -la "$OUT"
