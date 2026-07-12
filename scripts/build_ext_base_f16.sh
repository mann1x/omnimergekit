#!/usr/bin/env bash
# T87 llama.cpp anchor — build F16 GGUFs for ext (extended, proportional-base rope)
# and base. NO quantize: the anchor isolates rope fidelity, so F16 avoids any quant
# confound (and matches the bf16 vLLM anchor). YaRN is applied at llama-server
# runtime, not baked in.
set -euo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
CONVERT=/srv/ml/tools/llama.cpp/convert_hf_to_gguf.py
MERGED=/srv/ml/longctx/gemma-4-26B-A4B-it-512k          # extended (proportional_yarn cfg)
BASE=/srv/ml/google/gemma-4-26B-A4B-it                  # base (proportional cfg, native)
CDIR=/srv/ml/longctx/ext_convert                        # symlink-farm w/ patched config
OUT=/srv/ml/longctx/t87_llama
PATCH=/srv/ml/scripts/t87_patch_proportional_config.py
mkdir -p "$OUT"

# --- ext: symlink-farm the merged dir, swap in a proportional-base config ---
echo "[$(date +%T)] staging ext convert dir (patched proportional config)"
rm -rf "$CDIR"; mkdir -p "$CDIR"
for f in "$MERGED"/*; do ln -s "$f" "$CDIR/$(basename "$f")"; done
rm -f "$CDIR/config.json"
"$PY" "$PATCH" "$MERGED/config.json" "$CDIR/config.json"

echo "[$(date +%T)] convert EXT -> F16"
"$PY" "$CONVERT" "$CDIR" --outfile "$OUT/ext-f16.gguf" --outtype f16
echo "[$(date +%T)] EXT F16 done: $(ls -la "$OUT/ext-f16.gguf" | awk '{print $5}')"

echo "[$(date +%T)] convert BASE -> F16"
"$PY" "$CONVERT" "$BASE" --outfile "$OUT/base-f16.gguf" --outtype f16
echo "[$(date +%T)] BASE F16 done: $(ls -la "$OUT/base-f16.gguf" | awk '{print $5}')"

echo "[$(date +%T)] ALL DONE"
ls -la "$OUT"/*.gguf
