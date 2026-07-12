#!/usr/bin/env bash
# gen_t223_fk_maps.sh — generate the T223 force-keep drop maps (fkbroad UNION T223
# loop pins) at K={8,16,27}, validating EACH map against intent immediately after
# creation. CPU-only, fast. Reuses the EXACT gen_v8_fkbroad.sh recipe so the only
# change vs v8coder_fkbroad is the added --force-keep pins.
#
# Step 0 is a no-pin (broad-only) REPRO gate: regenerate with just the broad pins and
# confirm it == v8coder_fkbroad_drop_map.json set-for-set, proving the generator has
# not drifted before we trust its force-keep output on the new pins.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
GEN=/srv/ml/scripts/generate_drop_map_v5fk.py
EMIT=/srv/ml/scripts/emit_broad_fk.py
VAL=/srv/ml/scripts/validate_fk_map.py
DATA=/mnt/sdc/ml/google/expert_neuron_v7_code.json
FLOORDATA=/mnt/sdc/ml/google/expert_neuron_base_v7.json
FLOORMAP=/srv/ml/repos/omnimergekit/scripts/v4_layer_floor_map_v7.json
BASESEL=/srv/ml/scripts/teacher_force_98e_p16_clean.json
FKFILE=/srv/ml/scripts/broad_fk.txt
BASEMAP=/srv/ml/scripts/v8coder_fkbroad_drop_map.json
OUTDIR=/srv/ml/scripts
CLASSES="generic_math generic_logic generic_code generic_science generic_creative generic_multilingual targeted_humaneval targeted_humanevalplus targeted_lcb_medium_55"
COMMON=(--data "$DATA" --target 98 --protect-top 16 --alpha 2.0 --score-mode legacy
        --strategy max --normalize rank --classes $CLASSES --protect-strategy same
        --breadth-bonus 0.5 --v4-floor-data "$FLOORDATA" --v4-floor-map "$FLOORMAP"
        --outlier-wnorm-thresh 10000 --outlier-mode median --baseline-drop-map "$BASESEL")
WEIGHTS=(--class-weights 1 1 3 1 1 0 0 0 2)
ts(){ date '+%T %Z'; }

# ── T223 loop pins (from expert_neuron_v8_agentic_diff.json) ──────────────────
PINS_fk8="0:62,1:114,3:68,21:111,8:0,0:100,7:63,19:116"
PINS_fk16="3:68,22:29,1:114,18:102,20:0,7:63,8:85,25:122,7:44,11:22,8:0,0:62,0:100,5:85,19:116,21:111"
PINS_fk27="3:68,22:29,15:117,22:1,1:114,18:102,20:0,7:63,16:60,8:85,8:30,25:122,7:44,13:20,11:22,24:61,19:74,8:0,0:62,3:71,0:100,5:85,19:116,21:107,21:111,20:68,6:117"

for f in "$GEN" "$VAL" "$DATA" "$BASEMAP"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
[ -f "$FKFILE" ] || { echo "[gen $(ts)] broad_fk.txt missing -> emit"; "$PY" "$EMIT" >/dev/null || exit 3; }
BROAD="$(cat "$FKFILE")"
NBROAD=$(awk -F, '{print NF}' <<<"$BROAD")
echo "[gen $(ts)] broad pins: $NBROAD"

RC=0

# ── Step 0: REPRO gate (broad-only must reproduce the base fkbroad map) ───────
REPRO="$OUTDIR/_fkbroad_repro_t223gate.json"
echo "[gen $(ts)] === STEP 0: broad-only repro gate ==="
"$PY" "$GEN" "${COMMON[@]}" "${WEIGHTS[@]}" --force-keep "$BROAD" --output "$REPRO" 2>&1 | tail -3 \
  || { echo "[gen] FATAL repro generate"; exit 4; }
"$PY" "$VAL" "$BASEMAP" "$REPRO" "" "$BROAD" "REPRO-GATE(==base)" \
  || { echo "[gen] FATAL generator drifted — broad-only != base fkbroad. STOP."; exit 5; }

# ── Steps 1-3: the three force-keep maps ─────────────────────────────────────
for V in fk8 fk16 fk27; do
  eval "T=\$PINS_$V"
  NT=$(awk -F, '{print NF}' <<<"$T")
  OUT="$OUTDIR/v8coder_${V}_drop_map.json"
  echo "[gen $(ts)] === $V: generate (broad ${NBROAD} UNION ${NT} T223 pins) -> $OUT ==="
  "$PY" "$GEN" "${COMMON[@]}" "${WEIGHTS[@]}" --force-keep "$BROAD,$T" --output "$OUT" 2>&1 | tail -5 \
    || { echo "[gen] FATAL generate $V"; RC=1; continue; }
  echo "[gen $(ts)] --- validate $V against intent ---"
  "$PY" "$VAL" "$BASEMAP" "$OUT" "$T" "$BROAD" "$V" || { echo "[gen] VALIDATION FAILED $V"; RC=1; }
done

echo "[gen $(ts)] DONE rc=$RC"
exit $RC
