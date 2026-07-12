#!/bin/bash
# T176 Phase 2 — multilingual-aware re-rank sweep on the REBUILT competence maps.
#
# Inputs (Phase 1, bs2):
#   $G/expert_neuron_base_v6.json   (base map, 6 Tier-A cats incl generic_multilingual,
#                                    fp32-guarded producer) — drives the v4-floor protection
#   $G/expert_neuron_v6_code.json   (targeted map, 6 generic + 3 targeted code cats) —
#                                    drives the class-weighted drop RANKING
#
# Pipeline per ML weight w ∈ {0.5,1.0,2.0} (protective weight on generic_multilingual):
#   1. regen floor map from base map (top98_mean, guarded scorer)  [once]
#   2. gate both maps (finite + multilingual non-degeneracy)        [once]
#   3. drop map @ A2 recipe with new maps + class-weights 1 1 3 1 1 w 0 0 2
#   4. pristine bf16 62e (expert_drop only)
#   5. loop_screen.py on the SAME T175 200-prompt sample → multilingual loop rate
#   6. table vs A2 (3% anchor / 15.5% screen) + pristine-62e (45% screen) from T175
#
# A drop in the multilingual screen vs A2 is the headline T176 result. The winning
# w is then carried (separately) to: 62e + PES α=1.20 + imatrix Q6_K + full Stage-B
# audit (HE+164 / ifeval_100 / multipl_e_100 + loop) vs the A2 anchors.
#
# Recipe is the LOCKED A2 recipe (62e-reproduce bit-identical), with --data/--v4-floor-data
# swapped to the v6 maps and --v4-floor-map to the regenerated floor. Weights mapped by
# category NAME (not position) and asserted against the 9 expected categories.
set -uo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
G=/mnt/sdc/ml/google
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
SAMPLE=/mnt/sdc/ml/corpora/loop_screen_sample.jsonl
RES=/srv/ml/eval_results_tracks_2_3/t176_phase2
LOG=/srv/ml/logs/t176
mkdir -p "$RES" "$LOG"

BASE_MAP=$G/expert_neuron_base_v6.json
TGT_MAP=$G/expert_neuron_v6_code.json
FLOOR=$SCR/v4_layer_floor_map_v6.json
BASELINE=$SCR/teacher_force_98e_p16_clean.json
A2=$G/gemma-4-A4B-62e-fc15_25-p8-pes120-it
P62=$G/gemma-4-A4B-62e-fc15_25-p8-pristine-it

ML_WEIGHTS="${1:-0.5 1.0 2.0}"

echo "==================== T176 Phase2 $(date -Iseconds) ===================="

# 0 preflight
for f in "$BASE_MAP" "$TGT_MAP" "$SAMPLE" "$BASELINE"; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 2; }
done
[ -d "$SRC" ] || { echo "FATAL: 128e source missing $SRC"; exit 2; }
[ -d "$A2" ]  || echo "WARN: A2 anchor missing ($A2) — screen table will lack 3% anchor"

# 1 regen floor map from the rebuilt base map (guarded top98_mean, alpha matches recipe)
if [ ! -f "$FLOOR" ]; then
  echo "[1 $(date +%H:%M:%S)] regen floor map -> $FLOOR"
  $PY "$SCR/regen_floor_map.py" --base-map "$BASE_MAP" --alpha 2.0 \
    --outlier-wnorm-thresh 10000.0 --outlier-mode median \
    --out "$FLOOR" 2>&1 | tee "$LOG/floor_regen.log"
  [ -f "$FLOOR" ] || { echo "FATAL: floor regen failed"; exit 3; }
else
  echo "[1] floor map exists, skip"
fi

# 2 gate both maps (HARD finite check; multilingual non-degeneracy report)
echo "[2 $(date +%H:%M:%S)] gate base map"
$PY "$SCR/gate_competence_map.py" --map "$BASE_MAP" 2>&1 | tee "$LOG/gate_base.log"
gate_base=${PIPESTATUS[0]}
echo "[2 $(date +%H:%M:%S)] gate targeted map"
$PY "$SCR/gate_competence_map.py" --map "$TGT_MAP" 2>&1 | tee "$LOG/gate_tgt.log"
gate_tgt=${PIPESTATUS[0]}
if [ "$gate_base" -ne 0 ] || [ "$gate_tgt" -ne 0 ]; then
  echo "FATAL: a competence map FAILED the finite gate (base=$gate_base tgt=$gate_tgt) — STOP"
  exit 4
fi

# helper: emit the class-weight string for a given ML weight, by category NAME,
# asserting the 9 expected categories are exactly present (order-independent).
emit_weights(){ local mlw=$1
  $PY - "$TGT_MAP" "$mlw" <<'PY'
import json, sys
mp, mlw = sys.argv[1], float(sys.argv[2])
cats = json.load(open(mp))["metadata"]["categories"]
wmap = {"generic_math":1.0,"generic_logic":1.0,"generic_code":3.0,
        "generic_science":1.0,"generic_creative":1.0,"generic_multilingual":mlw,
        "targeted_humaneval":0.0,"targeted_humanevalplus":0.0,"targeted_lcb_medium_55":2.0}
expected = set(wmap)
got = set(cats)
if got != expected:
    sys.stderr.write(f"CATEGORY MISMATCH\n expected={sorted(expected)}\n got     ={sorted(got)}\n")
    sys.exit(9)
print(" ".join(str(wmap[c]) for c in cats))   # positional, in the map's own order
PY
}

# 3-5 per ML weight: drop map -> pristine 62e -> screen
declare -a SCREENED
for w in $ML_WEIGHTS; do
  tag=$(echo "$w" | sed 's/\./p/')           # 0.5 -> 0p5
  name="v6ml${tag}"
  DROP=$SCR/v6base_ml${tag}_62e_fc15_25_p8_drop_map.json
  OUT=$G/gemma-4-A4B-62e-v6ml${tag}-pristine-it
  SCREEN=$RES/${name}.json

  echo "==================== ML weight=$w (tag=$tag) ===================="
  CW=$(emit_weights "$w") || { echo "FATAL: weight build failed for ml=$w"; exit 9; }
  echo "[w=$w] class-weights = $CW"

  # 3 drop map (LOCKED A2 recipe, v6 maps + guarded floor)
  if [ ! -f "$DROP" ]; then
    echo "[3 $(date +%H:%M:%S)] drop map -> $DROP"
    $PY "$SCR/generate_drop_map_v5.py" \
      --data "$TGT_MAP" --target 62 --protect-top 8 --alpha 2.0 \
      --strategy max --normalize rank --class-weights $CW \
      --v4-floor-data "$BASE_MAP" --v4-floor-top 0 --v4-floor-clamp 15 25 \
      --v4-floor-map "$FLOOR" --breadth-bonus 0.5 \
      --baseline-drop-map "$BASELINE" \
      --outlier-wnorm-thresh 10000.0 --outlier-mode median \
      --output "$DROP" 2>&1 | tee "$LOG/dropmap_${name}.log" | tail -20
    [ -f "$DROP" ] || { echo "FATAL: drop map $name failed"; exit 5; }
  else
    echo "[3] drop map $name exists, skip"
  fi

  # 4 pristine bf16 62e
  if [ ! -f "$OUT/model.safetensors.index.json" ]; then
    echo "[4 $(date +%H:%M:%S)] expert_drop -> $OUT"
    $PY "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$DROP" \
      --output-dir "$OUT" 2>&1 | tee "$LOG/drop_${name}.log" | tail -4
    [ -f "$OUT/model.safetensors.index.json" ] || { echo "FATAL: build $name failed"; exit 6; }
  else
    echo "[4] 62e $name exists, skip"
  fi

  # 5 loop screen (GPU0; one at a time — keeps it simple, ~5-8 min each)
  if [ ! -f "$SCREEN" ]; then
    echo "[5 $(date +%H:%M:%S)] loop screen $name"
    CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error \
      $PY "$SCR/loop_screen.py" --model "$OUT" --name "$name" --sample "$SAMPLE" \
      --out "$SCREEN" >>"$LOG/screen_${name}.log" 2>&1 || echo "[5] screen FAIL $name"
  else
    echo "[5] screen $name exists, skip"
  fi
  SCREENED+=("$name=$SCREEN")
done

# 6 table vs A2 + pristine-62e (pull T175 anchors if present)
echo "==================== T176 Phase2 RESULT $(date +%H:%M:%S) ===================="
T175=/srv/ml/eval_results_tracks_2_3/t175_loop_screen
$PY - "$T175" "$RES" "${SCREENED[@]}" <<'PY'
import json, os, sys, glob
t175, res = sys.argv[1], sys.argv[2]
screened = dict(x.split("=", 1) for x in sys.argv[3:])
rows = {}
# anchors from T175 (a2-pes120, p62)
for f in glob.glob(os.path.join(t175, "*.json")):
    try:
        d = json.load(open(f)); rows[d["name"]] = d
    except Exception:
        pass
for name, path in screened.items():
    if os.path.exists(path):
        rows[name] = json.load(open(path))
order = ["a2-pes120", "p62"] + list(screened.keys())
print("%-12s %7s %9s   multilingual    constrained" % ("variant", "loop%", "loops/n"))
for k in order:
    if k not in rows:
        continue
    d = rows[k]; bb = d.get("by_bucket", {})
    ml = bb.get("multilingual", {}); co = bb.get("constrained", {})
    print("%-12s %6.1f%% %9s   %s   %s" % (
        k, d["loop_pct"], "%d/%d" % (d["loops"], d["n"]),
        "%d/%d" % (ml.get("loops", 0), ml.get("n", 0)),
        "%d/%d" % (co.get("loops", 0), co.get("n", 0))))
a2 = rows.get("a2-pes120", {}).get("loop_pct")
if a2:
    print("\nA2 screen=%.1f%% ↔ 3%% full-bench. Target <1%% full-bench ⇒ screen ≲ %.1f%%." % (a2, a2/3.0))
    for k in screened:
        if k in rows:
            lp = rows[k]["loop_pct"]
            verdict = "PASS proxy" if lp <= a2/3.0 else ("BELOW A2" if lp < a2 else "no better")
            print("  %-10s %.1f%%  %s" % (k, lp, verdict))
PY
echo "==================== T176 Phase2 DONE $(date +%H:%M:%S) ===================="
