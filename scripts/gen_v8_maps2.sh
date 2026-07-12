#!/usr/bin/env bash
# gen_v8_maps2.sh — floor-lever candidates (the real lever; weight-only A was
# floor-locked). All share the C6v3lcb recipe; differ in code/LCB weights +
# v4-generalist-floor clamp (frees slots that code/LCB then fill by aggregate).
#   B: code6 lcb5 + protect-class-min generic_code:2 (no floor change)
#   C: code5 lcb4 + v4-floor-clamp 12 20  (gentle: mean 16.9 -> ~14)
#   D: code6 lcb5 + v4-floor-clamp 8 16   (strong: mean 16.9 -> ~10)
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
GEN=/srv/ml/scripts/generate_drop_map_v5.py
DATA=/mnt/sdc/ml/google/expert_neuron_v7_code.json
FLOORDATA=/mnt/sdc/ml/google/expert_neuron_base_v7.json
FLOORMAP=/srv/ml/repos/omnimergekit/scripts/v4_layer_floor_map_v7.json
BASE=/srv/ml/scripts/teacher_force_98e_p16_clean.json
OUT=/srv/ml/scripts
CLASSES="generic_math generic_logic generic_code generic_science generic_creative generic_multilingual targeted_humaneval targeted_humanevalplus targeted_lcb_medium_55"
COMMON=(--data "$DATA" --target 98 --protect-top 16 --alpha 2.0 --score-mode legacy
        --strategy max --normalize rank --classes $CLASSES --protect-strategy same
        --breadth-bonus 0.5 --v4-floor-data "$FLOORDATA" --v4-floor-map "$FLOORMAP"
        --outlier-wnorm-thresh 10000 --outlier-mode median --baseline-drop-map "$BASE")
ts(){ date '+%T %Z'; }

echo "[gen2 $(ts)] B) code6 lcb5 + protect-class-min generic_code:2"
"$PY" "$GEN" "${COMMON[@]}" --class-weights 1 1 6 1 1 0 0 0 5 \
  --protect-class-min generic_code:2 \
  --output "$OUT/v8coder_code6lcb5_cmin2_drop_map.json" 2>&1 | tail -3 \
  || echo "[gen2] WARN B failed"

echo "[gen2 $(ts)] C) code5 lcb4 + v4-floor-clamp 12 20"
"$PY" "$GEN" "${COMMON[@]}" --class-weights 1 1 5 1 1 0 0 0 4 \
  --v4-floor-clamp 12 20 \
  --output "$OUT/v8coder_code5lcb4_fc12_20_drop_map.json" 2>&1 | tail -3 \
  || echo "[gen2] WARN C failed"

echo "[gen2 $(ts)] D) code6 lcb5 + v4-floor-clamp 8 16"
"$PY" "$GEN" "${COMMON[@]}" --class-weights 1 1 6 1 1 0 0 0 5 \
  --v4-floor-clamp 8 16 \
  --output "$OUT/v8coder_code6lcb5_fc8_16_drop_map.json" 2>&1 | tail -3 \
  || echo "[gen2] WARN D failed"

echo "[gen2 $(ts)] DONE"; ls -la "$OUT"/v8coder_*_drop_map.json 2>/dev/null
