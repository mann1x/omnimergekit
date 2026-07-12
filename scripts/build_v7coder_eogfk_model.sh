#!/usr/bin/env bash
# build_v7coder_eogfk_model.sh — T202.4 model build (CPU only).
#
# Faithful v7-coder recipe with ONLY the drop-map changed (12 force-keep
# terminator experts). Single-variable vs the published C6v3lcb model.
#   expert_drop(eog_fk map) -> router_shared_upweight(alpha=1.2) ->
#   convert F16 (llama.cpp-latest) -> quantize Q4_K_M (llama.cpp-latest, no imatrix).
# Matched toolchain == the gate's llama-server (llama.cpp-latest).
set -euo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
LCPP=/mnt/sdc/ml/llama.cpp-latest
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
DROP=/mnt/sdc/ml/sft_heal/v7coder_eog_fk_drop_map.json

CAND=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-eogfk-it      # bf16 dir
F16=/mnt/sdc/ml/sft_heal/v7coder-eogfk-F16.gguf
Q4=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-eogfk-it-Q4_K_M.gguf

ts(){ date '+%T %Z'; }
echo "==================== build eogfk $(ts) ===================="

for f in "$SRC/config.json" "$DROP" "$LCPP/convert_hf_to_gguf.py" "$LCPP/build/bin/llama-quantize"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done

# ── 1. expert_drop ─────────────────────────────────────────
if [ ! -f "$CAND/model.safetensors.index.json" ]; then
  echo "[1 $(ts)] expert_drop -> $CAND"
  "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$DROP" --output-dir "$CAND"
  [ -f "$CAND/model.safetensors.index.json" ] || { echo "FATAL expert_drop failed"; exit 3; }
  [ -f "$CAND/tokenizer.json" ] || { echo "FATAL tokenizer.json not copied"; exit 3; }
else
  echo "[1] $CAND exists, skip expert_drop"
fi

# ── 2. router_shared_upweight (v7-coder recipe: alpha=1.2, bf16 target) ──
if [ ! -f "$CAND/.shared_applied" ]; then
  echo "[2 $(ts)] router_shared_upweight --alpha 1.2 --target mlp.down_proj.weight"
  "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$CAND" \
    --alpha 1.2 --target mlp.down_proj.weight
  # router_shared_upweight.py does NOT write the marker itself; we record it on
  # success so the convert step (and idempotent re-runs) never re-scale (alpha^2).
  touch "$CAND/.shared_applied"
else
  echo "[2] .shared_applied exists, skip"
fi

# ── 3. convert to F16 GGUF ─────────────────────────────────
if [ ! -f "$F16" ]; then
  echo "[3 $(ts)] convert -> $F16"
  "$PY" "$LCPP/convert_hf_to_gguf.py" "$CAND" --outfile "$F16" --outtype f16
  [ -f "$F16" ] || { echo "FATAL convert failed"; exit 5; }
  echo "  F16 size: $(du -h "$F16" | cut -f1)"
else
  echo "[3] $F16 exists, skip convert"
fi

# ── 4. quantize Q4_K_M (no imatrix — matches baseline single-variable) ──
if [ ! -f "$Q4" ]; then
  echo "[4 $(ts)] quantize Q4_K_M -> $Q4"
  "$LCPP/build/bin/llama-quantize" "$F16" "$Q4" Q4_K_M
  [ -f "$Q4" ] || { echo "FATAL quantize failed"; exit 6; }
  echo "  Q4 size: $(du -h "$Q4" | cut -f1)"
fi

# ── 5. drop F16 intermediate (CLAUDE.md: delete GGUF intermediates) ──
echo "[5 $(ts)] rm F16 intermediate $F16"
rm -f "$F16"

echo
echo "==================== BUILD DONE $(ts) ===================="
echo "candidate Q4: $Q4"
ls -la "$Q4"
sha256sum "$Q4" | awk '{print "sha256:", $1}'
echo "[gate-ready] run: bash /srv/ml/repos/omnimergekit/recipes/gemma4/agentic_loop/run_gate_rkdeog.sh $Q4"
