#!/usr/bin/env bash
# build_qat_pruned_base.sh — regenerate the v7-coder QAT-pruned bf16 dir + F16 GGUF
# (with the published v7 shared-alpha=1.2 step), the SHARED artifact for the v7
# QAT low-bit / CD-qat investigation (items 2/3/4, 2026-06-07).
#
# Unlike build_v7_qat.sh (which casts straight to Q4_0 and `rm -rf`s the dir+F16),
# this KEEPS the bf16 dir and F16 so we can cut MANY tiers off it:
#   - qat-Q2_K_L / qat-Q3_K_M (raw low-bit from QAT base — item 4 vs vanilla sweep)
#   - CD-qat-Q4_K_M (v7 fixed CD map + protection on QAT base — item 2, corrected)
#   - CD-qat low-floor (item 3)
# Recipe is byte-faithful to the published v7-coder (shared a=1.2 on down_proj) so
# the comparison vs the vanilla sweep tiers (also shared-a) isolates the QAT base.
#
# CPU-only (expert_drop / shared / convert). No GPU contention with the sweep.
set -uo pipefail
BM=/srv/ml
PY="$BM/envs/envs/omnimergekit/bin/python"
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"
SCR="$BM/scripts"
CONVERT="$BM/tools/llama.cpp/convert_hf_to_gguf.py"
QAT_BASE=/mnt/sdc/ml/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
MAP="$SCR/v7coder_g15f2440_drop_map.json"
PUB_DIR=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it
OUT=/mnt/sdc/ml/qat_investig
DIR="$OUT/v7coder-qat-pruned"          # shared-alpha bf16 dir (KEPT)
F16="$OUT/v7coder-qat-F16.gguf"        # F16 (KEPT)
mkdir -p "$OUT"
LOG="$OUT/build_qat_pruned_base_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
L(){ echo "[qatbase $(date -u +%H:%M:%S)] $*"; }

L "=== regenerate v7-coder QAT-pruned bf16 + F16 (shared a=1.2) ==="
for f in "$PY" "$SCR/expert_drop.py" "$SCR/router_shared_upweight.py" "$CONVERT" "$MAP" "$QAT_BASE" "$PUB_DIR"; do
  [ -e "$f" ] || { L "FATAL missing $f"; exit 1; }
done
[ -f "$QAT_BASE/model.safetensors.index.json" ] || { L "FATAL QAT base incomplete"; exit 1; }
DPL=$("$PY" -c "import json,statistics as s;d=json.load(open('$MAP'));v=[len(x) for x in d.values() if isinstance(x,list)];print(int(s.mean(v)) if v else -1)")
L "drop map dropped/layer=$DPL (expect 30 -> 98e)"
[ "$DPL" = "30" ] || { L "FATAL drop map not 98e (dpl=$DPL)"; exit 1; }

# [1] expert_drop (QAT base + v7-coder map)
if [ -f "$DIR/model.safetensors.index.json" ]; then
  L "[1] expert_drop dir exists, skip"
else
  L "[1] expert_drop QAT_base -> $DIR"
  "$PY" "$SCR/expert_drop.py" --source-dir "$QAT_BASE" --drop-map "$MAP" --output-dir "$DIR" 2>&1 | tail -6
  [ -f "$DIR/model.safetensors.index.json" ] || { L "FATAL expert_drop failed"; exit 1; }
fi
for aux in preprocessor_config.json processor_config.json generation_config.json tokenizer.json tokenizer_config.json; do
  if [ ! -f "$DIR/$aux" ] && [ -f "$PUB_DIR/$aux" ]; then cp -n "$PUB_DIR/$aux" "$DIR/$aux"; L "  aux $aux"; fi
done

# [2] shared alpha=1.2 (v7 recipe — MANDATORY, matches published bf16)
if [ ! -f "$DIR/.shared_applied" ]; then
  L "[2] router_shared_upweight --alpha 1.2 --target mlp.down_proj.weight"
  "$PY" "$SCR/router_shared_upweight.py" --model-dir "$DIR" --alpha 1.2 --target mlp.down_proj.weight 2>&1 | tail -6
  rm -f "$DIR"/*.pre_shared_upweight
  touch "$DIR/.shared_applied"
else
  L "[2] .shared_applied present, skip"
fi

# [3] convert F16 (KEPT)
if [ -s "$F16" ]; then
  L "[3] F16 exists, skip ($(du -h "$F16"|cut -f1))"
else
  L "[3] convert -> F16"
  "$PY" "$CONVERT" "$DIR" --outfile "$F16" --outtype f16 2>&1 | tail -3
  n=$("$PY" -c "from gguf.gguf_reader import GGUFReader;print(len(GGUFReader('$F16').tensors))" 2>/dev/null)
  L "  F16 tensors=$n"; [ "${n:-0}" -lt 600 ] && { L "FATAL too few tensors ($n)"; exit 1; }
fi

L "=== DONE: bf16=$DIR  F16=$(du -h "$F16"|cut -f1) ==="
L "###### QAT_PRUNED_BASE_DONE ######"
