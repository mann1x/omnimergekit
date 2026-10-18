#!/usr/bin/env bash
# build_k12_map.sh — v10 candidate: map A + 12 force-keep rescues (owner 2026-10-17: "go with k12, build the map
# and show the swaps"). Same generator call as recipes/gemma4/v9_agentic/reproduce_drop_map.sh; only the
# rescue pins change:  the 8 shipped ml_k8 rescues + 4 cross-class hub experts map A drops
# (8:75 17:71 2:65 3:61 — top-4 in generic_code and targeted_swe_java, top-7 multilingual, top-5 ifeval).
# hub_k9 (owner 2026-10-17: "build also the 2nd map without multi-lingual"): the same minus the 3
# multilingual-only rescues 23:116 26:16 29:25 (rank 97-111 in generic_code) = 5 hub rescues of k8 + HUB4.
# Usage (from the omnimergekit repo root): PY=/path/to/python bash recipes/gemma4/v10_agentic/build_k12_map.sh <outdir>
set -euo pipefail
PY=${PY:-python3}
R=recipes/gemma4/v9_agentic
OUT=${1:?outdir}
mkdir -p "$OUT"
BASE=$R/competence/expert_neuron_v9_agentic_code.slim.json
FLOOR=$R/competence/expert_neuron_base_v7.slim.json
LIFT=$(cat $R/maps/candidates/eog_v4_t106_lift_pins.txt)
P12="0:62,1:114,3:68,21:111,8:0,0:100,7:63,19:116,8:85,18:102,25:122,5:85"
K8="8:122,15:106,27:48,29:25,26:16,15:64,0:9,23:116"
HUB4="8:75,17:71,2:65,3:61"
HUB5="8:122,15:106,27:48,15:64,0:9"
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
gen ml_k8_repro "$LIFT,$P12,$K8"
cmp "$OUT/ml_k8_repro.json" $R/maps/gemma-4-20B-v9-agentic_drop_map.json || { echo "FAIL: shipped ml_k8 not reproduced"; exit 1; }
echo "control: shipped ml_k8 reproduced byte-identical"
gen hub_k12 "$LIFT,$P12,$K8,$HUB4"
echo "hub_k12 -> $OUT/hub_k12.json ($(sha256sum "$OUT/hub_k12.json" | cut -c1-16))"
gen hub_k9 "$LIFT,$P12,$HUB5,$HUB4"
echo "hub_k9 -> $OUT/hub_k9.json ($(sha256sum "$OUT/hub_k9.json" | cut -c1-16))"
