#!/usr/bin/env bash
# reproduce_drop_map.sh — regenerate the gemma-4-20B-v9-agentic drop map from the data in this repo.
#
#   1. map A        = the agentic competence map + 132 force-keep pins (steps 1–5 of the recipe)
#   2. ladder pins  = map A's dropped experts ranked on generic_multilingual / generic_ifeval
#   3. ml_k8        = map A + the 8 multilingual rescues (step 6) = the published drop map
#
# Every step is gated: map A must equal maps/candidates/A_lift120_plus_fk12.json, the pins must
# equal maps/v9ml_ladder/pins.txt, and ml_k8 must be byte-identical to
# maps/gemma-4-20B-v9-agentic_drop_map.json (sha256 84f2bd961772adbd…). CPU only, ~1 min.
# The competence maps are the slim forms (neuron_act stripped; the generator never reads it).
#
# Usage (from the repo root):  PY=/path/to/python bash recipes/gemma4/v9_agentic/reproduce_drop_map.sh [outdir]
set -euo pipefail
PY=${PY:-python3}
R=recipes/gemma4/v9_agentic
OUT=${1:-$(mktemp -d)}
[ -f "$R/README.md" ] || { echo "run from the omnimergekit repo root"; exit 1; }
mkdir -p "$OUT"
BASE=$R/competence/expert_neuron_v9_agentic_code.slim.json      # map A's data (Tier A + Tier B)
MERGED=$R/competence/expert_neuron_v9ml_code.slim.json          # + generic_multilingual, generic_ifeval
FLOOR=$R/competence/expert_neuron_base_v7.slim.json             # v4-floor ranking data
A_REF=$R/maps/candidates/A_lift120_plus_fk12.json
SHIPPED=$R/maps/gemma-4-20B-v9-agentic_drop_map.json
LIFT=$(cat $R/maps/candidates/eog_v4_t106_lift_pins.txt)        # 120 end-of-turn lift pins
P12="0:62,1:114,3:68,21:111,8:0,0:100,7:63,19:116,8:85,18:102,25:122,5:85"   # 12 loop pins

mapfile -t CW < <("$PY" scripts/v9_classes.py "$BASE")
CLASSES="${CW[0]}"; WEIGHTS="${CW[1]}"

gen() { # name force-keep
  # shellcheck disable=SC2086
  "$PY" recipes/gemma4/generate_drop_map_v5fk.py --data "$BASE" --target 98 --protect-top 16 --alpha 2.0 \
    --score-mode legacy --strategy max --normalize rank --classes $CLASSES --class-weights $WEIGHTS \
    --protect-strategy same --class-protect-floor 0 \
    --v4-floor-data "$FLOOR" --v4-floor-top 0 --v4-floor-map scripts/v4_layer_floor_map_v7.json \
    --breadth-bonus 0.5 --outlier-wnorm-thresh 10000.0 --outlier-mode median \
    --baseline-drop-map scripts/teacher_force_98e_p16_clean.json --force-keep "$2" \
    --output "$OUT/$1.json" > "$OUT/$1.log" 2>&1
}

gen A "$LIFT,$P12"
"$PY" scripts/v9ml_same_map.py "$OUT/A.json" "$A_REF" || { echo "FAIL: map A not reproduced"; exit 1; }
echo "map A reproduced"

"$PY" scripts/v9ml_ladder.py pins "$MERGED" "$A_REF" > "$OUT/pins.txt"
diff -q "$OUT/pins.txt" $R/maps/v9ml_ladder/pins.txt || { echo "FAIL: ladder pins differ"; exit 1; }
echo "ladder pins reproduced"

PINS=$(awk '$1=="ml_k8"{print $2}' "$OUT/pins.txt")
gen ml_k8 "$LIFT,$P12,$PINS"
cmp "$OUT/ml_k8.json" "$SHIPPED" || { echo "FAIL: ml_k8 differs from the published drop map"; exit 1; }
echo "ml_k8 byte-identical to $SHIPPED ($(sha256sum "$SHIPPED" | cut -c1-16)) — outputs in $OUT"
