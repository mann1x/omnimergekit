#!/usr/bin/env bash
# build_v7coder_gsweep.sh — fs2440 + targeted_gpqa weight/floor sweep.
# Args: SUFFIX GPQA_WEIGHT FLOOR_LO FLOOR_HI
# IDENTICAL to build_v7coder_fs2440g.sh except targeted_gpqa weight + floor clamp.
set -uo pipefail
SUF="${1:?suffix}"; W="${2:?gpqa weight}"; FLO="${3:?floor lo}"; FHI="${4:?floor hi}"
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"
SCR=$BM/scripts
GEN=$SCR/generate_drop_map_v5.py
DATA=/mnt/sdc/ml/google/expert_neuron_v7_code_gpqa.json
FLOOR=$BM/repos/omnimergekit/scripts/v4_layer_floor_map_v7.json
FLOORDATA=/mnt/sdc/ml/google/expert_neuron_base_v7.json
BASELINE=$SCR/teacher_force_98e_p16_clean.json
DROP=$SCR/v7coder_${SUF}_drop_map.json
SRC_HF=$BM/models/base/gemma-4-26B-A4B-it
OUT=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-${SUF}-it
GG=${OUT}-GGUF
F16=$GG/gemma-4-A4B-98e-v7-coder-${SUF}-it-F16.gguf
Q6=$GG/gemma-4-A4B-98e-v7-coder-${SUF}-it-Q6_K.gguf
CONVERT=$BM/tools/llama.cpp/convert_hf_to_gguf.py
QUANT=/opt/llama.cpp/build/bin/llama-quantize
NAME=v7coder-${SUF}-q6k
RES=$BM/eval_results_v7coder_${SUF}
mkdir -p "$GG" "$RES"
L(){ echo "[gsweep:$SUF $(date -u +%H:%M:%S)] $*"; }

L "=== preflight (suffix=$SUF weight=$W floor=$FLO-$FHI) ==="
for f in "$GEN" "$DATA" "$FLOOR" "$FLOORDATA" "$BASELINE" "$SCR/expert_drop.py" "$SCR/router_shared_upweight.py" "$SRC_HF" "$CONVERT" "$QUANT"; do
  [ -e "$f" ] || { L "FATAL missing: $f"; exit 1; }
done
AVAIL=$(df -BG --output=avail /mnt/sdc/ml | tail -1 | tr -dc 0-9)
L "disk avail: ${AVAIL}G"
[ "${AVAIL:-0}" -lt 70 ] && { L "FATAL <70G free"; exit 1; }
"$PY" -c "import json;c=json.load(open(\"$DATA\"))[\"categories\"];assert \"targeted_gpqa\" in c;print(\"map ok\")" || { L "FATAL map missing targeted_gpqa"; exit 1; }

if [ -f "$DROP" ]; then L "[1] drop exists, skip"; else
  L "[1] generate_drop_map_v5 (gpqa weight=$W, floor $FLO-$FHI)"
  "$PY" "$GEN" \
    --data "$DATA" --target 98 \
    --protect-top 16 --alpha 2.0 --strategy max --normalize rank \
    --breadth-bonus 0.5 --v4-floor-map "$FLOOR" --v4-floor-data "$FLOORDATA" \
    --v4-floor-clamp "$FLO" "$FHI" \
    --baseline-drop-map "$BASELINE" \
    --outlier-mode median --outlier-wnorm-thresh 1e4 \
    --classes generic_math generic_logic generic_code generic_science generic_creative generic_multilingual targeted_humaneval targeted_humanevalplus targeted_lcb_medium_55 targeted_gpqa \
    --class-weights 1 1 3 1 1 0 0 0 2 "$W" \
    --output "$DROP" --summary-output "$DROP.summary.json" 2>&1 | tail -10
  [ -f "$DROP" ] || { L "FATAL drop not written"; exit 1; }
fi
DPL=$("$PY" -c "import json,statistics as s;d=json.load(open(\"$DROP\"));v=[len(x) for x in d.values() if isinstance(x,list)];print(int(s.mean(v)) if v else -1)")
L "[1] dropped/layer=$DPL"
[ "$DPL" = "30" ] || { L "FATAL not 98e (dpl=$DPL)"; exit 1; }

if [ -f "$OUT/model.safetensors.index.json" ]; then L "[2] bf16 exists, skip"; else
  L "[2] expert_drop -> $OUT"
  "$PY" "$SCR/expert_drop.py" --source-dir "$SRC_HF" --drop-map "$DROP" --output-dir "$OUT" 2>&1 | tail -6
  [ -f "$OUT/model.safetensors.index.json" ] || { L "FATAL expert_drop failed"; exit 1; }
fi

if [ -f "$OUT/.shared_applied" ]; then L "[3] shared applied, skip"; else
  L "[3] router_shared_upweight alpha=1.2"
  "$PY" "$SCR/router_shared_upweight.py" --model-dir "$OUT" --alpha 1.2 --target mlp.down_proj.weight 2>&1 | tail -4
  rm -f "$OUT"/*.pre_shared_upweight
  touch "$OUT/.shared_applied"
fi

if [ -f "$Q6" ]; then L "[4] Q6 exists, skip"; else
  L "[4] convert F16"
  "$PY" "$CONVERT" "$OUT" --outfile "$F16" --outtype f16 2>&1 | tail -3
  n=$("$PY" -c "from gguf.gguf_reader import GGUFReader; print(len(GGUFReader(\"$F16\").tensors))")
  L "[4] F16 tensors=$n"; [ "$n" -lt 600 ] && { L "FATAL too few tensors"; exit 1; }
  L "[4] quantize Q6_K"
  "$QUANT" "$F16" "$Q6" Q6_K 2>&1 | tail -3
  rm -f "$F16"
fi
ls -la "$Q6"
L "BUILD_DONE $SUF $Q6"
