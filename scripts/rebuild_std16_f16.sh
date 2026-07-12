#!/usr/bin/env bash
# rebuild_std16_f16.sh — regenerate STD16 bf16 + F16 (deleted by campaign disk-hygiene).
# Byte-faithful to the eval'd STD16: same recipe as build_t223_fk_campaign.sh build_cell STD16
#   expert_drop(v8coder_fk16_drop_map) -> router_shared_upweight a=1.2 -> convert f16 (no DERN).
# CPU-ONLY (no CUDA) -> safe to run while the GPUs serve the 12B harness.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
LCPP=/mnt/sdc/ml/llama.cpp-latest
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
WORK=/mnt/sdc/ml/t223_fk
MAP=/srv/ml/scripts/v8coder_fk16_drop_map.json
BF=$WORK/STD16-bf16
F16=$WORK/STD16-F16.gguf
ts(){ date '+%T %Z'; }

echo "[std16-rebuild $(ts)] START (CPU-only)"
for f in "$SRC/config.json" "$SCR/expert_drop.py" "$RECIPE/router_shared_upweight.py" \
         "$MAP" "$LCPP/convert_hf_to_gguf.py"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
echo "[std16-rebuild $(ts)] disk:"; df -h "$WORK" | tail -1

if [ -f "$F16" ]; then echo "[std16-rebuild $(ts)] F16 exists, skip"; echo "STD16_F16_REBUILD_DONE"; exit 0; fi

if [ ! -f "$BF/.shared_applied" ]; then
  rm -rf "$BF"
  echo "[std16-rebuild $(ts)] expert_drop (CPU) -> $BF"
  "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$MAP" --output-dir "$BF" \
    || { echo "FATAL expert_drop"; exit 1; }
  [ -f "$BF/tokenizer.json" ] || { echo "FATAL no tokenizer in $BF"; exit 1; }
  echo "[std16-rebuild $(ts)] router_shared_upweight a=1.2 (CPU)"
  "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$BF" \
    --alpha 1.2 --target mlp.down_proj.weight || { echo "FATAL shared"; exit 1; }
  touch "$BF/.shared_applied"
else
  echo "[std16-rebuild $(ts)] bf16 student already present (.shared_applied), skip drop/shared"
fi

echo "[std16-rebuild $(ts)] convert F16 (CPU)"
"$PY" "$LCPP/convert_hf_to_gguf.py" "$BF" --outfile "$F16" --outtype f16 \
  || { echo "FATAL convert"; exit 1; }
magic=$("$PY" -c "import sys; sys.stdout.write(open('$F16','rb').read(4).decode('latin1'))" 2>/dev/null)
[ "$magic" = "GGUF" ] || { echo "FATAL bad F16 GGUF magic='$magic'"; exit 1; }
echo "[std16-rebuild $(ts)] F16 DONE: $(stat -c%s "$F16") bytes -> $F16"
echo "[std16-rebuild $(ts)] bf16 kept at $BF (needed for bf16 repo + NVFP4A16)"
echo "STD16_F16_REBUILD_DONE"
