#!/usr/bin/env bash
# gen_v8_fkbroad.sh — the force-keep candidate. The weight/floor levers (A/B/C/D)
# could not rescue code/LCB (diff_v8_rescue proved they pull in generalists). The
# working lever is --force-keep: pin the 30 BROAD code/LCB losses directly. This
# is the C6v3lcb recipe verbatim (class-weights 1 1 3 1 1 0 0 0 2) + a hard pin
# list emitted by emit_broad_fk.py.
#
# The deployed bs2 generator (39,429 B, Jun 1) predates --force-keep. We use the
# newer canonical generator deployed as generate_drop_map_v5fk.py and GATE on a
# no-pin repro proving it still reproduces C6v3lcb set-for-set before trusting
# its --force-keep output. CPU-only, fast.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
GEN=/srv/ml/scripts/generate_drop_map_v5fk.py     # NEW deployed canonical (has --force-keep)
EMIT=/srv/ml/scripts/emit_broad_fk.py
DIFF=/srv/ml/scripts/diff_v8_rescue.py
VALEQ=/srv/ml/scripts/validate_dropmap_eq.py
DATA=/mnt/sdc/ml/google/expert_neuron_v7_code.json
FLOORDATA=/mnt/sdc/ml/google/expert_neuron_base_v7.json
FLOORMAP=/srv/ml/repos/omnimergekit/scripts/v4_layer_floor_map_v7.json
BASE=/srv/ml/scripts/teacher_force_98e_p16_clean.json
REF=/srv/ml/scripts/v7coder_C6v3lcb_drop_map.json   # reference selection (published v7-coder)
FKFILE=/srv/ml/scripts/broad_fk.txt
REPRO=/srv/ml/scripts/_c6_repro_fkgen.json
OUT=/srv/ml/scripts/v8coder_fkbroad_drop_map.json
CLASSES="generic_math generic_logic generic_code generic_science generic_creative generic_multilingual targeted_humaneval targeted_humanevalplus targeted_lcb_medium_55"
COMMON=(--data "$DATA" --target 98 --protect-top 16 --alpha 2.0 --score-mode legacy
        --strategy max --normalize rank --classes $CLASSES --protect-strategy same
        --breadth-bonus 0.5 --v4-floor-data "$FLOORDATA" --v4-floor-map "$FLOORMAP"
        --outlier-wnorm-thresh 10000 --outlier-mode median --baseline-drop-map "$BASE")
WEIGHTS=(--class-weights 1 1 3 1 1 0 0 0 2)
ts(){ date '+%T %Z'; }

[ -f "$GEN" ] || { echo "[fk] FATAL $GEN not deployed"; exit 9; }
grep -q -- "force.keep\|force_keep" "$GEN" || { echo "[fk] FATAL $GEN has no force-keep"; exit 9; }

echo "[fk $(ts)] 0) GATE: no-pin repro of C6v3lcb with the force-keep generator"
"$PY" "$GEN" "${COMMON[@]}" "${WEIGHTS[@]}" --output "$REPRO" 2>&1 | tail -3 \
  || { echo "[fk] FATAL repro generate"; exit 1; }
"$PY" "$VALEQ" "$REPRO" "$REF" || { echo "[fk] FATAL repro != C6v3lcb — generator drifted, STOP"; exit 2; }

echo "[fk $(ts)] 1) emit broad code/LCB pins"
"$PY" "$EMIT" 2>&1 | tail -4 || { echo "[fk] FATAL emit"; exit 3; }
PINS="$(cat "$FKFILE")"
NPIN=$(awk -F, '{print NF}' <<<"$PINS")
echo "[fk $(ts)] pins=$NPIN"
[ -z "$PINS" ] && { echo "[fk] FATAL empty pin list"; exit 4; }

echo "[fk $(ts)] 2) generate force-keep map (C6v3lcb recipe + --force-keep)"
"$PY" "$GEN" "${COMMON[@]}" "${WEIGHTS[@]}" --force-keep "$PINS" --output "$OUT" 2>&1 | tail -10 \
  || { echo "[fk] FATAL generate fkbroad"; exit 5; }

echo "[fk $(ts)] 3) rescue diff vs C6v3lcb"
"$PY" "$DIFF" "$OUT" 2>&1 | tail -12 || echo "[fk] WARN diff"

echo "[fk $(ts)] DONE"; ls -la "$OUT" "$FKFILE"
