#!/usr/bin/env bash
# gen_v8_maps.sh — regenerate C6v3lcb (repro/validate flags) then emit two
# code/LCB-protective candidate maps for the v8-coder loop-fix re-selection.
#   A: weight bump only        (generic_code 3->5, targeted_lcb 2->4)
#   B: stronger bump + code floor (code 6, lcb 5, --protect-class-min code:2)
# Everything else identical to the C6v3lcb recipe. CPU-only, fast.
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

echo "[gen $(ts)] 0) REPRO C6v3lcb (validate reconstructed flags)"
"$PY" "$GEN" "${COMMON[@]}" --class-weights 1 1 3 1 1 0 0 0 2 \
  --output "$OUT/v7coder_C6v3lcb_REPRO.json" 2>&1 | tail -4 \
  || { echo "[gen] FATAL repro generate"; exit 1; }

echo "[gen $(ts)] A) weight bump code5 lcb4"
"$PY" "$GEN" "${COMMON[@]}" --class-weights 1 1 5 1 1 0 0 0 4 \
  --output "$OUT/v8coder_code5lcb4_drop_map.json" 2>&1 | tail -4 \
  || { echo "[gen] FATAL candidate A"; exit 2; }

echo "[gen $(ts)] B) bump code6 lcb5 + protect-class-min code:2"
"$PY" "$GEN" "${COMMON[@]}" --class-weights 1 1 6 1 1 0 0 0 5 \
  --protect-class-min code:2 \
  --output "$OUT/v8coder_code6lcb5_floor2_drop_map.json" 2>&1 | tail -6 \
  || echo "[gen] WARN candidate B failed (likely protect-class-min name) — A still valid"

echo "[gen $(ts)] DONE; outputs:"
ls -la "$OUT"/v7coder_C6v3lcb_REPRO.json "$OUT"/v8coder_code5lcb4_drop_map.json "$OUT"/v8coder_code6lcb5_floor2_drop_map.json 2>/dev/null
