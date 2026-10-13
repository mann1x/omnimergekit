#!/usr/bin/env bash
# gen_fk12_map.sh — generate v8coder_fk12_drop_map.json: the K=12 rung of the T223
# force-keep ladder, for the STD12 cell (v9-agentic still loops at STD16; STD27 was
# worse in real use, so the next question is BELOW 16, not above it).
#
# THE K=12 PIN SET IS DERIVED, NOT INVENTED. The existing ladder hardcodes three pin
# strings in gen_t223_fk_maps.sh with no stated selection rule. Recovered it from
# /mnt/sdc/ml/google/expert_neuron_v8_agentic_diff.json (40 candidates, each with
# loop_mass / clean_mass / specificity = loop/clean) and PROVED it against all three
# published rungs before using it:
#
#   top-8  by specificity  ==  PINS_fk8   set-for-set   (exact)
#   top-16 by specificity  ==  PINS_fk16  set-for-set   (exact)
#   specificity >= 1.5     ==  PINS_fk27  set-for-set   (exact)
#
# So fkK = top-K by specificity, and K=12 is the 12 highest-specificity candidates:
#   rank  pin      spec        rank  pin      spec
#      1  0:62     2.97           7  7:63     2.13
#      2  1:114    2.86           8  19:116   2.08
#      3  3:68     2.81           9  8:85     2.02
#      4  21:111   2.65          10  18:102   1.99
#      5  8:0      2.25          11  25:122   1.97
#      6  0:100    2.24          12  5:85     1.95
# Added over fk8: 8:85, 18:102, 25:122, 5:85. Strictly nested: fk8 < fk12 < fk16.
#
# Everything else is byte-identical to gen_t223_fk_maps.sh: same generator, same
# COMMON/WEIGHTS, same broad pin file, and the same STEP 0 repro gate — regenerate
# with ONLY the broad pins and require it to equal v8coder_fkbroad_drop_map.json
# set-for-set, so a drifted generator is caught before its force-keep output is trusted.
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
DIFF=/mnt/sdc/ml/google/expert_neuron_v8_agentic_diff.json
OUTDIR=/srv/ml/scripts
CLASSES="generic_math generic_logic generic_code generic_science generic_creative generic_multilingual targeted_humaneval targeted_humanevalplus targeted_lcb_medium_55"
# shellcheck disable=SC2206  # $CLASSES MUST word-split into 9 separate --classes args;
# this array is byte-faithful to gen_t223_fk_maps.sh and must not be "improved".
COMMON=(--data "$DATA" --target 98 --protect-top 16 --alpha 2.0 --score-mode legacy
        --strategy max --normalize rank --classes $CLASSES --protect-strategy same
        --breadth-bonus 0.5 --v4-floor-data "$FLOORDATA" --v4-floor-map "$FLOORMAP"
        --outlier-wnorm-thresh 10000 --outlier-mode median --baseline-drop-map "$BASESEL")
WEIGHTS=(--class-weights 1 1 3 1 1 0 0 0 2)
ts(){ date '+%T %Z'; }

PINS_fk8="0:62,1:114,3:68,21:111,8:0,0:100,7:63,19:116"
PINS_fk16="3:68,22:29,1:114,18:102,20:0,7:63,8:85,25:122,7:44,11:22,8:0,0:62,0:100,5:85,19:116,21:111"
PINS_fk12="0:62,1:114,3:68,21:111,8:0,0:100,7:63,19:116,8:85,18:102,25:122,5:85"

for f in "$GEN" "$VAL" "$DATA" "$BASEMAP" "$DIFF"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done

# ── STEP -1: re-derive the rule from the data, here, every run ────────────────
# The pin string above is a cached result. Recompute it and refuse to proceed if
# the data no longer reproduces the two published rungs — a selection rule that
# cannot be re-derived is a guess, whatever it says in a comment.
echo "[gen $(ts)] === STEP -1: re-derive fkK = top-K by specificity ==="
"$PY" - "$DIFF" "$PINS_fk8" "$PINS_fk16" "$PINS_fk12" <<'PYEOF' || exit 3
import json, sys
diff, s8, s16, s12 = sys.argv[1:5]
rows = json.load(open(diff))["dropped_loop_expert_candidates"]
ranked = sorted(enumerate(rows),
                key=lambda t: (-t[1]["specificity"], -t[1]["loop_mass"], t[0]))
top = lambda k: {f"{r['layer']}:{r['expert']}" for _, r in ranked[:k]}
for k, s, name in ((8, s8, "fk8"), (16, s16, "fk16")):
    want = set(s.split(","))
    if top(k) != want:
        sys.exit(f"FATAL: top-{k} by specificity != {name}. "
                 f"only-in-top={sorted(top(k)-want)} only-in-{name}={sorted(want-top(k))}. "
                 f"The ladder rule does not hold on this data — STOP.")
    print(f"  top-{k:<2} == {name}  (exact, {k} pins)")
want12 = set(s12.split(","))
if top(12) != want12:
    sys.exit(f"FATAL: cached fk12 != top-12. computed={sorted(top(12))} cached={sorted(want12)}")
print(f"  top-12 == cached PINS_fk12 (exact)")
assert set(s8.split(",")) < want12 < set(s16.split(",")), "fk8 < fk12 < fk16 nesting broken"
print("  nesting fk8 < fk12 < fk16: OK")
PYEOF

[ -f "$FKFILE" ] || { echo "[gen $(ts)] broad_fk.txt missing -> emit"; "$PY" "$EMIT" >/dev/null || exit 4; }
BROAD="$(cat "$FKFILE")"
NBROAD=$(awk -F, '{print NF}' <<<"$BROAD")
echo "[gen $(ts)] broad pins: $NBROAD"

# ── STEP 0: REPRO gate (broad-only must reproduce the base fkbroad map) ───────
REPRO="$OUTDIR/_fkbroad_repro_fk12gate.json"
echo "[gen $(ts)] === STEP 0: broad-only repro gate ==="
"$PY" "$GEN" "${COMMON[@]}" "${WEIGHTS[@]}" --force-keep "$BROAD" --output "$REPRO" 2>&1 | tail -3 \
  || { echo "[gen] FATAL repro generate"; exit 5; }
"$PY" "$VAL" "$BASEMAP" "$REPRO" "" "$BROAD" "REPRO-GATE(==base)" \
  || { echo "[gen] FATAL generator drifted — broad-only != base fkbroad. STOP."; exit 6; }

# ── STEP 1: the fk12 map ─────────────────────────────────────────────────────
NT=$(awk -F, '{print NF}' <<<"$PINS_fk12")
OUT="$OUTDIR/v8coder_fk12_drop_map.json"
echo "[gen $(ts)] === fk12: generate (broad ${NBROAD} UNION ${NT} T223 pins) -> $OUT ==="
"$PY" "$GEN" "${COMMON[@]}" "${WEIGHTS[@]}" --force-keep "$BROAD,$PINS_fk12" --output "$OUT" 2>&1 | tail -5 \
  || { echo "[gen] FATAL generate fk12"; exit 7; }
echo "[gen $(ts)] --- validate fk12 against intent ---"
"$PY" "$VAL" "$BASEMAP" "$OUT" "$PINS_fk12" "$BROAD" "fk12" \
  || { echo "[gen] VALIDATION FAILED fk12"; exit 8; }

# ── STEP 2: fk12 must differ from fk8 by EXACTLY the 4 added pins ────────────
# The realised DROP MAPS do NOT nest, and asserting they do was wrong: the budget
# is fixed at 30 dropped per layer, so force-keeping a 9th..12th pin forces 4
# COMPENSATING EVICTIONS to stay in budget. Every rung drops exactly 900 experts;
# a larger K is not a subset, it is a swap. What IS nested is the PIN set, and the
# invariant worth gating is therefore: relative to fk8, fk12 rescues exactly the 4
# pins that top-12 adds over top-8, and evicts exactly 4 to pay for them.
echo "[gen $(ts)] === STEP 2: fk12 vs fk8 = exactly the 4 added pins, 4-for-4 ==="
"$PY" - "$OUTDIR/v8coder_fk8_drop_map.json" "$OUT" "$PINS_fk8" "$PINS_fk12" <<'PYEOF' || exit 9
import json, sys
def dropped(p):
    d = json.load(open(p)); d = d.get("drop_map", d)
    return {(int(k), e) for k in d if str(k).lstrip("-").isdigit() for e in d[k]}
f8p, f12p, s8, s12 = sys.argv[1:5]
f8, f12 = dropped(f8p), dropped(f12p)
pin = lambda s: {(int(a), int(b)) for a, b in (x.split(":") for x in s.split(","))}
added = pin(s12) - pin(s8)
rescued = f8 - f12          # dropped by fk8, kept by fk12
evicted = f12 - f8          # kept by fk8, dropped by fk12
print(f"  budget: fk8 drops {len(f8)}, fk12 drops {len(f12)} (fixed by design)")
print(f"  pins added over fk8 : {sorted(added)}")
print(f"  rescued by fk12     : {sorted(rescued)}")
print(f"  evicted to pay      : {sorted(evicted)}")
if rescued != added:
    sys.exit(f"FATAL: fk12 rescued {sorted(rescued)} but top-12 adds {sorted(added)} "
             f"over top-8 — the map does not implement the requested rung.")
if len(evicted) != len(added):
    sys.exit(f"FATAL: {len(added)} pins rescued but {len(evicted)} evicted — "
             f"budget accounting is off.")
if evicted & pin(s12):
    sys.exit(f"FATAL: an eviction hit a forced pin: {sorted(evicted & pin(s12))}")
print("  rescue set == added pins, 4-for-4, no forced pin evicted: OK")
PYEOF

echo "[gen $(ts)] === FK12_MAP_DONE rc=0 -> $OUT ==="
