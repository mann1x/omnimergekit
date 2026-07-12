#!/usr/bin/env bash
# gate_dern11.sh — T203: convert the DERN-Eq.11 merged bf16 -> Q4_K_M (no imatrix,
# matched llama.cpp-latest toolchain) and run the 12-seed agentic gate.
#
# The merged student already carries the v7-coder shared alpha=1.2 (we merged the
# published student, which has .shared_applied), so there is NO router_shared step.
# Single-variable vs the fresh-rebuild 5/12 no-swap anchor + the published 11/12.
set -euo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
LCPP=/mnt/sdc/ml/llama.cpp-latest
CAND=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-dern11-it
F16=/mnt/sdc/ml/sft_heal/v7coder-dern11-F16.gguf
Q4=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-dern11-it-Q4_K_M.gguf

ts(){ date '+%T %Z'; }
echo "==================== gate_dern11 $(ts) ===================="
# weights: single-shard model.safetensors OR sharded model.safetensors.index.json.
# save_pretrained writes a single shard (no index) when the model fits one file,
# which is the DERN-Eq.11 merge case (40.9 GB safetensors, no index.json).
[ -e "$CAND/model.safetensors" ] || [ -e "$CAND/model.safetensors.index.json" ] || {
  echo "FATAL missing weights (model.safetensors or .index.json) in $CAND"; exit 2; }
for f in "$CAND/tokenizer.json" "$CAND/config.json" \
         "$LCPP/convert_hf_to_gguf.py" "$LCPP/build/bin/llama-quantize"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done

if [ ! -f "$Q4" ]; then
  if [ ! -f "$F16" ]; then
    echo "[1 $(ts)] convert -> $F16"
    "$PY" "$LCPP/convert_hf_to_gguf.py" "$CAND" --outfile "$F16" --outtype f16
    [ -f "$F16" ] || { echo "FATAL convert failed"; exit 3; }
  fi
  echo "[2 $(ts)] quantize Q4_K_M -> $Q4"
  "$LCPP/build/bin/llama-quantize" "$F16" "$Q4" Q4_K_M
  [ -f "$Q4" ] || { echo "FATAL quantize failed"; exit 4; }
  echo "  Q4 size: $(du -h "$Q4" | cut -f1)"
  rm -f "$F16"
else
  echo "[1-2] $Q4 exists, skip build"
fi

echo "[3 $(ts)] sha256: $(sha256sum "$Q4" | awk '{print $1}')"
echo "[4 $(ts)] launching 12-seed agentic gate vs published baseline"
bash /srv/ml/agentic_loop/run_gate_rkdeog.sh "$Q4"
echo "==================== gate_dern11 DONE $(ts) ===================="
