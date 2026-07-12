#!/usr/bin/env bash
# build_v7coder.sh — tentative 98e v7-coder.
# Recipe = PUBLISHED v6-coder C6v3lcb knobs, byte-identical, fed the NEW v7
# fixed maps (T176 rebuild): --data expert_neuron_v7_code.json + the v7-rebuild
# per-layer floor v4_layer_floor_map_v7.json. Only the competence map + floor
# change vs v6-coder; every selection knob is held fixed.
#   target=98, protect-top=16, alpha=2.0, strategy=max, normalize=rank,
#   breadth-bonus=0.5, outlier median@1e4, class-weights 1 1 3 1 1 0 0 2
#   MANDATORY post: router_shared_upweight --alpha 1.2 --target mlp.down_proj.weight
# Quant: PLAIN Q6_K (matches build_v6coder_C6v3lcb.sh -> HE+164 0.9329 anchor).
# Eval: HE+164 (humanevalplus_full) + MPE-100 (multipl_e_100), llama backend.
# Anchors (v6-coder C6v3lcb Q6_K): HE+164 0.9329 / MPE-100 0.89.
set -uo pipefail
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"
SCR=$BM/scripts
GEN=$SCR/generate_drop_map_v5.py
DATA=/mnt/sdc/ml/google/expert_neuron_v7_code.json
FLOOR=$BM/repos/omnimergekit/scripts/v4_layer_floor_map_v7.json
FLOORDATA=/mnt/sdc/ml/google/expert_neuron_base_v7.json
BASELINE=$SCR/teacher_force_98e_p16_clean.json
DROP=$SCR/v7coder_fs3450_drop_map.json
SRC_HF=$BM/models/base/gemma-4-26B-A4B-it
OUT=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs3450-it
GG=${OUT}-GGUF
F16=$GG/gemma-4-A4B-98e-v7-coder-fs3450-it-F16.gguf
Q6=$GG/gemma-4-A4B-98e-v7-coder-fs3450-it-Q6_K.gguf
CONVERT=$BM/tools/llama.cpp/convert_hf_to_gguf.py
QUANT=/opt/llama.cpp/build/bin/llama-quantize
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
NAME=v7coder-fs3450-q6k
RES=$BM/eval_results_v7coder_fs3450
GPU="${1:-0}"
mkdir -p "$GG" "$RES"
L(){ echo "[v7coder $(date -u +%H:%M:%S)] $*"; }

for f in "$GEN" "$DATA" "$FLOOR" "$FLOORDATA" "$BASELINE" "$SCR/expert_drop.py" "$SCR/router_shared_upweight.py" "$SRC_HF" "$CONVERT" "$QUANT" "$OMK"; do
  [ -e "$f" ] || { L "FATAL missing: $f"; exit 1; }
done
L "=== v7-coder build (v6 C6v3lcb knobs + v7 maps) ==="

# --- [1] drop map ---
if [ -f "$DROP" ]; then L "[1] drop map exists, skip"; else
  L "[1] generate_drop_map_v5 (v7 code map + v7 floor, C6v3lcb knobs)"
  "$PY" "$GEN" \
    --data "$DATA" --target 98 \
    --protect-top 16 --alpha 2.0 --strategy max --normalize rank \
    --breadth-bonus 0.5 --v4-floor-map "$FLOOR" --v4-floor-data "$FLOORDATA" \
    --v4-floor-clamp 34 50 \
    --baseline-drop-map "$BASELINE" \
    --outlier-mode median --outlier-wnorm-thresh 1e4 \
    --classes generic_math generic_logic generic_code generic_science generic_creative generic_multilingual targeted_humaneval targeted_humanevalplus targeted_lcb_medium_55 \
    --class-weights 1 1 3 1 1 0 0 0 2 \
    --output "$DROP" --summary-output "$DROP.summary.json" 2>&1 | tail -12
  [ -f "$DROP" ] || { L "FATAL drop map not written"; exit 1; }
fi
# verify 98e (drop 30/layer)
DPL=$("$PY" -c "import json,statistics as s;d=json.load(open('$DROP'));v=[len(x) for x in d.values() if isinstance(x,list)];print(int(s.mean(v)) if v else -1)")
L "[1] dropped/layer=$DPL (expect 30 -> 98e)"
[ "$DPL" = "30" ] || { L "FATAL not 98e (dropped/layer=$DPL)"; exit 1; }

# --- [2] expert_drop ---
if [ -f "$OUT/model.safetensors.index.json" ]; then L "[2] bf16 exists, skip"; else
  L "[2] expert_drop.py -> $OUT"
  "$PY" "$SCR/expert_drop.py" --source-dir "$SRC_HF" --drop-map "$DROP" --output-dir "$OUT" 2>&1 | tail -8
  [ -f "$OUT/model.safetensors.index.json" ] || { L "FATAL expert_drop failed"; exit 1; }
fi

# --- [3] MANDATORY shared alpha=1.2 ---
if [ -f "$OUT/.shared_applied" ]; then L "[3] .shared_applied exists, skip"; else
  L "[3] router_shared_upweight --alpha 1.2 --target mlp.down_proj.weight"
  "$PY" "$SCR/router_shared_upweight.py" --model-dir "$OUT" --alpha 1.2 --target mlp.down_proj.weight 2>&1 | tail -6
  rm -f "$OUT"/*.pre_shared_upweight
  touch "$OUT/.shared_applied"
fi

# --- [4] convert F16 + plain Q6_K ---
if [ -f "$Q6" ]; then L "[4] Q6_K exists, skip"; else
  L "[4] convert bf16 -> F16"
  "$PY" "$CONVERT" "$OUT" --outfile "$F16" --outtype f16 2>&1 | tail -4
  n=$("$PY" -c "from gguf.gguf_reader import GGUFReader; print(len(GGUFReader('$F16').tensors))")
  L "[4] F16 tensors=$n"; [ "$n" -lt 600 ] && { L "FATAL too few tensors"; exit 1; }
  L "[4] quantize plain Q6_K"
  "$QUANT" "$F16" "$Q6" Q6_K 2>&1 | tail -4
  rm -f "$F16"
fi
ls -la "$Q6"

ls -la "$Q6"
touch "${Q6}.BUILD_DONE"
L "BUILD_DONE fs3450 $Q6"
