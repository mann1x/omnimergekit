#!/usr/bin/env bash
# build_v7coder_noswap_control.sh — T202.4 TOOLCHAIN CONTROL (CPU only).
#
# Identical pipeline to build_v7coder_eogfk_model.sh, but with the ORIGINAL
# C6v3lcb drop-map (NO force-keep). This isolates the toolchain/metadata effect
# from the 12-expert swap:
#   - candidate (eogfk) gated 4/12 vs the DOWNLOADED published baseline 11/12.
#   - the candidate is a FRESH llama.cpp-latest convert; the baseline carries the
#     known special-token metadata bug. So 11->4 conflates toolchain + swap.
#   - this control = no-swap C6v3lcb through the SAME fresh pipeline. Gate it vs
#     the same published baseline:
#       ~4/12  -> toolchain/metadata fix caused it, swap did nothing.
#       ~11/12 -> the 12-expert swap genuinely cut looping (lever works).
set -euo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
LCPP=/mnt/sdc/ml/llama.cpp-latest
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
DROP=/srv/ml/repos/omnimergekit/scripts/v7coder_C6v3lcb_drop_map.json   # ORIGINAL, no force-keep

CAND=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-noswap-mytc-it
F16=/mnt/sdc/ml/sft_heal/v7coder-noswap-mytc-F16.gguf
Q4=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-noswap-mytc-it-Q4_K_M.gguf

ts(){ date '+%T %Z'; }
echo "==================== build noswap-control $(ts) ===================="
for f in "$SRC/config.json" "$DROP" "$LCPP/convert_hf_to_gguf.py" "$LCPP/build/bin/llama-quantize"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done

if [ ! -f "$CAND/model.safetensors.index.json" ]; then
  echo "[1 $(ts)] expert_drop (C6v3lcb, no force-keep) -> $CAND"
  "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$DROP" --output-dir "$CAND"
  [ -f "$CAND/model.safetensors.index.json" ] || { echo "FATAL expert_drop failed"; exit 3; }
  [ -f "$CAND/tokenizer.json" ] || { echo "FATAL tokenizer.json not copied"; exit 3; }
else echo "[1] $CAND exists, skip"; fi

if [ ! -f "$CAND/.shared_applied" ]; then
  echo "[2 $(ts)] router_shared_upweight --alpha 1.2 --target mlp.down_proj.weight"
  "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$CAND" \
    --alpha 1.2 --target mlp.down_proj.weight
  touch "$CAND/.shared_applied"
else echo "[2] .shared_applied exists, skip"; fi

if [ ! -f "$F16" ]; then
  echo "[3 $(ts)] convert -> $F16"
  "$PY" "$LCPP/convert_hf_to_gguf.py" "$CAND" --outfile "$F16" --outtype f16
  [ -f "$F16" ] || { echo "FATAL convert failed"; exit 5; }
else echo "[3] $F16 exists, skip"; fi

if [ ! -f "$Q4" ]; then
  echo "[4 $(ts)] quantize Q4_K_M -> $Q4"
  "$LCPP/build/bin/llama-quantize" "$F16" "$Q4" Q4_K_M
  [ -f "$Q4" ] || { echo "FATAL quantize failed"; exit 6; }
  echo "  Q4 size: $(du -h "$Q4" | cut -f1)"
fi
echo "[5 $(ts)] rm F16 intermediate"; rm -f "$F16"

echo
echo "==================== NOSWAP-CONTROL BUILD DONE $(ts) ===================="
ls -la "$Q4"; sha256sum "$Q4" | awk '{print "sha256:", $1}'
echo "[gate-ready] bash /srv/ml/agentic_loop/run_gate_rkdeog.sh $Q4"
