#!/usr/bin/env bash
# v9-ml post-map chain: merge (basis-gated) -> reproduce map A -> class-weight sweep -> report.
# CPU only. Every arm = map A generator call verbatim; only --classes/--class-weights change.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=""
PY=/root/anaconda3/envs/omnimergekit/bin/python
[ -x "$PY" ] || PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11
H=/mnt/sdc/harness
G=/mnt/sdc/ml/google
BASE=$G/expert_neuron_v9_agentic_code.json
TIERA=$G/expert_neuron_v9ml_tiera.json
MERGED=$G/expert_neuron_v9ml_code.json
O=/mnt/sdc/ml/v9ml/maps; mkdir -p $O
A_REF=/mnt/sdc/ml/std12v5/candidates/A_lift120_plus_fk12.json
LIFT=$(cat $G/eog_v4_t106_lift_pins.txt)
P12="0:62,1:114,3:68,21:111,8:0,0:100,7:63,19:116,8:85,18:102,25:122,5:85"
FK="$LIFT,$P12"
die(){ echo "V9ML_POST_FAIL $*"; exit 1; }
echo "[post] start $(date -u +%FT%TZ)"
"$PY" /srv/ml/repos/omnimergekit/scripts/merge_tier_a_categories.py --base $BASE --add $TIERA --out $MERGED || die merge
mapfile -t CW < <("$PY" $H/v9_classes.py "$BASE"); CLASSES="${CW[0]}"; WEIGHTS="${CW[1]}"
[ -n "$CLASSES" ] || die classes
gen(){ # name data classes weights
  # shellcheck disable=SC2086
  "$PY" /srv/ml/scripts/gen_drop_v5_fk.py --data "$2" --target 98 --protect-top 16 --alpha 2.0 --score-mode legacy \
    --strategy max --normalize rank --classes $3 --class-weights $4 --protect-strategy same --class-protect-floor 0 \
    --v4-floor-data $G/expert_neuron_base_v7.json --v4-floor-top 0 --v4-floor-map /srv/ml/repos/omnimergekit/scripts/v4_layer_floor_map_v7.json \
    --breadth-bonus 0.5 --outlier-wnorm-thresh 10000.0 --outlier-mode median \
    --baseline-drop-map /srv/ml/scripts/teacher_force_98e_p16_clean.json --force-keep "${FK}" --output "${O}/$1.json" > "${O}/$1.log" 2>&1 || die "gen $1"
}
gen A_repro "$BASE" "$CLASSES" "$WEIGHTS"
"$PY" $H/v9ml_same_map.py "$O/A_repro.json" "$A_REF" || die "A not reproduced"
echo "[post] A_repro == delivered map A"
for arm in "ml0_if0 0 0" "ml1_if1 1 1" "ml2_if2 2 2" "ml2_if0 2 0" "ml0_if2 0 2"; do
  read -r name wml wif <<< "$arm"
  gen "$name" "$MERGED" "$CLASSES generic_multilingual generic_ifeval" "$WEIGHTS $wml $wif"
  echo "[post] built $name"
done
"$PY" $H/v9ml_report.py "$O" "$A_REF" "$MERGED" "$P12" || die report
# Minimal-swap ladder (owner 2026-10-15: fewest moves vs A = highest value). A's data, classes,
# weights and force-keep, plus K rescue pins ranked by shared multilingual+ifeval importance.
L=/mnt/sdc/ml/v9ml/ladder; mkdir -p $L
"$PY" $H/v9ml_ladder.py pins "$MERGED" "$A_REF" > $L/pins.txt || die "ladder pins"
cat $L/pins.txt
while read -r name pins; do
  O=$L FK="$LIFT,$P12,$pins" gen "$name" "$BASE" "$CLASSES" "$WEIGHTS"
  echo "[post] ladder built $name"
done < $L/pins.txt
"$PY" $H/v9ml_ladder.py report "$MERGED" "$A_REF" $L || die "ladder report"
echo "V9ML_POST_OK $(date -u +%FT%TZ)"
