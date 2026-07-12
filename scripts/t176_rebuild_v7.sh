#!/bin/bash
# T176.6 — ANSWER-CHANNEL full re-sweep on the de-confounded competence maps.
#
# Council csl-2026-06-01-0714-14c6 + code-verified two map bugs that made the
# v6 (thinking-channel) re-sweep monotonically inert:
#   (1) wnorm was a SUM over routed tokens, and the drop-map score added +tc on
#       top -> routing FREQUENCY double-counted -> rare-routed (low-resource
#       language) experts suppressed regardless of class weight.
#   (2) Tier-A generated with enable_thinking=True @128 tok -> the map profiled
#       the <think> CoT channel; foreign-language ANSWER production (where the
#       loops happen) was truncated/never reached.
#
# Fixes now live in the producer/consumer:
#   producer: wnorm -> per-token RMS sqrt(wnsq/tc); --no-tier-a-thinking generates
#             the ANSWER directly (answer-channel map).
#   consumer: --score-mode mean = wnorm*alpha (no +tc) for RMS maps.
#
# This script rebuilds BOTH maps answer-channel + RMS (v7), regenerates the
# floor with --score-mode mean, prints an informational keep-set diff vs the v6
# (thinking-channel) drop map as an early read, then runs the full
# ML in {0.5,1.0,2.0} drop-map -> pristine 62e -> loop-screen sweep.
#
# HEADLINE: does the de-confounded answer-channel signal move the multilingual
# loop rate where the thinking-channel v6 maps were inert?
#   - moves down  -> the selection lever was real, just mis-measured; build the
#                    winner + PES + Q6_K -> Stage-B audit vs A2 anchors.
#   - still inert -> 5th independent falsification of selection; commit SFT-heal.
#
# Recipe is the LOCKED A2 recipe, with v7 maps + --score-mode mean + v7 floor.
set -uo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
G=/mnt/sdc/ml/google
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
MLCALIB=/mnt/sdc/ml/corpora/multilingual_calib.jsonl
TIERB=/mnt/sdc/ml/corpora/v5_code_pass_traces.json
SAMPLE=/mnt/sdc/ml/corpora/loop_screen_sample.jsonl
BASELINE=$SCR/teacher_force_98e_p16_clean.json
RES=/srv/ml/eval_results_tracks_2_3/t176_phase3
LOG=/srv/ml/logs/t176
mkdir -p "$RES" "$LOG"

BASE_MAP=$G/expert_neuron_base_v7.json
TGT_MAP=$G/expert_neuron_v7_code.json
FLOOR=$SCR/v4_layer_floor_map_v7.json
V6_DROP=$SCR/v6base_ml1p0_62e_fc15_25_p8_drop_map.json   # thinking-channel ref
A2=$G/gemma-4-A4B-62e-fc15_25-p8-pes120-it
P62=$G/gemma-4-A4B-62e-fc15_25-p8-pristine-it
GPUB=90
TAMAX=512

ML_WEIGHTS="${1:-0.5 1.0 2.0}"

echo "==================== T176.6 rebuild-v7 $(date -Iseconds) ===================="

# 0 preflight
for f in "$MLCALIB" "$TIERB" "$SAMPLE" "$BASELINE"; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 2; }
done
[ -d "$SRC" ] || { echo "FATAL: 128e source missing $SRC"; exit 2; }

# 1 BASE map — Tier-A only, ANSWER channel, RMS wnorm
if [ ! -f "$BASE_MAP" ]; then
  echo "[1 $(date +%H:%M:%S)] BASE map (answer-channel) -> $BASE_MAP"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
    $PY "$SCR/expert_neuron_analysis_v5_targeted.py" \
      --variant base --model "$SRC" --gpu-budget-gib "$GPUB" \
      --multilingual-calib "$MLCALIB" \
      --no-tier-a-thinking --tier-a-max-tokens "$TAMAX" \
      --out "$BASE_MAP" 2>&1 | tee "$LOG/base_map_v7.log"
  [ -f "$BASE_MAP" ] || { echo "FATAL: base map v7 failed"; exit 3; }
else
  echo "[1] base map v7 exists, skip"
fi

# 2 TARGETED map — reuse base Tier-A (answer-channel), add Tier-B code replay
if [ ! -f "$TGT_MAP" ]; then
  echo "[2 $(date +%H:%M:%S)] TARGETED map -> $TGT_MAP"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
    $PY "$SCR/expert_neuron_analysis_v5_targeted.py" \
      --variant code --model "$SRC" --gpu-budget-gib "$GPUB" \
      --tier-b-json "$TIERB" --load-tier-a-from "$BASE_MAP" \
      --multilingual-calib "$MLCALIB" --no-tier-a-thinking \
      --out "$TGT_MAP" 2>&1 | tee "$LOG/targeted_map_v7.log"
  [ -f "$TGT_MAP" ] || { echo "FATAL: targeted map v7 failed"; exit 3; }
else
  echo "[2] targeted map v7 exists, skip"
fi

# 2.5 confirm wnorm_mode + answer-channel stamped in metadata
$PY - "$BASE_MAP" "$TGT_MAP" <<'PY'
import json, sys
for p in sys.argv[1:]:
    m = json.load(open(p))["metadata"]
    print(f"[meta] {p.split('/')[-1]}: wnorm_mode={m.get('wnorm_mode')} "
          f"tier_a_thinking={m.get('tier_a_thinking')} cats={len(m.get('categories', []))}")
    assert m.get("wnorm_mode") == "rms_per_token", "wnorm_mode not RMS!"
    assert m.get("tier_a_thinking") is False, "tier_a_thinking not False (answer-channel)!"
print("[meta] OK — both maps are RMS + answer-channel")
PY
[ $? -eq 0 ] || { echo "FATAL: map metadata wrong"; exit 3; }

# 3 gate both maps (finite + ML non-degeneracy)
for M in "$BASE_MAP" "$TGT_MAP"; do
  echo "[3 $(date +%H:%M:%S)] gate $(basename "$M")"
  $PY "$SCR/gate_competence_map.py" --map "$M" 2>&1 | tee "$LOG/gate_v7_$(basename "$M" .json).log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FATAL: gate FAIL $M"; exit 4; }
done

# 4 floor regen from base map (--score-mode mean — RMS scoring)
echo "[4 $(date +%H:%M:%S)] regen floor (mean) -> $FLOOR"
$PY "$SCR/regen_floor_map.py" --base-map "$BASE_MAP" --alpha 2.0 \
  --outlier-wnorm-thresh 10000.0 --outlier-mode median \
  --score-mode mean --out "$FLOOR" 2>&1 | tee "$LOG/floor_regen_v7.log"
[ -f "$FLOOR" ] || { echo "FATAL: floor regen v7 failed"; exit 4; }

# helper: class-weight string by category NAME, asserting the 9 expected cats.
emit_weights(){ local mlw=$1
  $PY - "$TGT_MAP" "$mlw" <<'PY'
import json, sys
mp, mlw = sys.argv[1], float(sys.argv[2])
cats = json.load(open(mp))["metadata"]["categories"]
wmap = {"generic_math":1.0,"generic_logic":1.0,"generic_code":3.0,
        "generic_science":1.0,"generic_creative":1.0,"generic_multilingual":mlw,
        "targeted_humaneval":0.0,"targeted_humanevalplus":0.0,"targeted_lcb_medium_55":2.0}
if set(cats) != set(wmap):
    sys.stderr.write(f"CATEGORY MISMATCH\n expected={sorted(wmap)}\n got={sorted(cats)}\n")
    sys.exit(9)
print(" ".join(str(wmap[c]) for c in cats))
PY
}

# 5 per-ML: drop map (mean) -> [diff vs v6 on ml=1.0] -> pristine 62e -> screen
declare -a SCREENED
for w in $ML_WEIGHTS; do
  tag=$(echo "$w" | sed 's/\./p/')
  name="v7ml${tag}"
  DROP=$SCR/v7base_ml${tag}_62e_fc15_25_p8_drop_map.json
  OUT=$G/gemma-4-A4B-62e-v7ml${tag}-pristine-it
  SCREEN=$RES/${name}.json

  echo "==================== ML weight=$w (tag=$tag) ===================="
  CW=$(emit_weights "$w") || { echo "FATAL: weight build failed ml=$w"; exit 9; }
  echo "[w=$w] class-weights = $CW"

  if [ ! -f "$DROP" ]; then
    echo "[5 $(date +%H:%M:%S)] drop map (mean) -> $DROP"
    $PY "$SCR/generate_drop_map_v5.py" \
      --data "$TGT_MAP" --target 62 --protect-top 8 --alpha 2.0 \
      --strategy max --normalize rank --class-weights $CW --score-mode mean \
      --v4-floor-data "$BASE_MAP" --v4-floor-top 0 --v4-floor-clamp 15 25 \
      --v4-floor-map "$FLOOR" --breadth-bonus 0.5 \
      --baseline-drop-map "$BASELINE" \
      --outlier-wnorm-thresh 10000.0 --outlier-mode median \
      --output "$DROP" 2>&1 | tee "$LOG/dropmap_${name}.log" | tail -20
    [ -f "$DROP" ] || { echo "FATAL: drop map $name failed"; exit 5; }
  else
    echo "[5] drop map $name exists, skip"
  fi

  # 5b DIFF GATE (informational early read): v7 keep set vs v6 thinking-channel
  if [ "$tag" = "1p0" ] && [ -f "$V6_DROP" ]; then
    echo "[5b $(date +%H:%M:%S)] keep-set diff v7ml1p0 vs v6ml1p0 (thinking-channel)"
    $PY - "$DROP" "$V6_DROP" <<'PY' 2>&1 | tee "$LOG/diff_v7_vs_v6_ml1p0.log"
import json, sys
def keepset(p):
    d = json.load(open(p)); dm = d.get("drop_map", d)
    out = {}
    for k, v in dm.items():
        if str(k).isdigit():
            out[int(k)] = set(range(128)) - set(int(x) for x in v)
    return out
a, b = keepset(sys.argv[1]), keepset(sys.argv[2])
layers = sorted(set(a) & set(b))
tot_swap = 0; jl = []
for li in layers:
    inter = len(a[li] & b[li]); uni = len(a[li] | b[li])
    swap = len(a[li] - b[li])
    tot_swap += swap
    jl.append(inter / uni if uni else 1.0)
import statistics as st
print(f"  layers={len(layers)} mean_jaccard={st.mean(jl):.3f} "
      f"min={min(jl):.3f}  total_experts_swapped(v7 not in v6)={tot_swap}")
print(f"  -> {'MATERIAL DIFF (answer-channel changed the keep set)' if tot_swap>=30 else 'NEAR-IDENTICAL (channel/RMS barely moved selection — expect inert)'}")
PY
  fi

  if [ ! -f "$OUT/model.safetensors.index.json" ]; then
    echo "[6 $(date +%H:%M:%S)] expert_drop -> $OUT"
    $PY "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$DROP" \
      --output-dir "$OUT" 2>&1 | tee "$LOG/drop_${name}.log" | tail -4
    [ -f "$OUT/model.safetensors.index.json" ] || { echo "FATAL: build $name failed"; exit 6; }
  else
    echo "[6] 62e $name exists, skip"
  fi

  if [ ! -f "$SCREEN" ]; then
    echo "[7 $(date +%H:%M:%S)] loop screen $name"
    CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error \
      $PY "$SCR/loop_screen.py" --model "$OUT" --name "$name" --sample "$SAMPLE" \
      --out "$SCREEN" >>"$LOG/screen_${name}.log" 2>&1 || echo "[7] screen FAIL $name"
  else
    echo "[7] screen $name exists, skip"
  fi
  SCREENED+=("$name=$SCREEN")
done

# 8 result table vs A2 + v6 anchors
echo "==================== T176.6 RESULT $(date +%H:%M:%S) ===================="
T175=/srv/ml/eval_results_tracks_2_3/t175_loop_screen
P2=/srv/ml/eval_results_tracks_2_3/t176_phase2
$PY - "$T175" "$P2" "${SCREENED[@]}" <<'PY'
import json, os, sys, glob
t175, p2 = sys.argv[1], sys.argv[2]
screened = dict(x.split("=", 1) for x in sys.argv[3:])
rows = {}
for d in (t175, p2):
    for f in glob.glob(os.path.join(d, "*.json")):
        try:
            j = json.load(open(f)); rows[j["name"]] = j
        except Exception:
            pass
for name, path in screened.items():
    if os.path.exists(path):
        rows[name] = json.load(open(path))
order = ["a2-pes120", "p62", "v6ml0p5", "v6ml1p0", "v6ml2p0"] + list(screened.keys())
print("%-12s %7s %9s   multilingual    constrained" % ("variant", "loop%", "loops/n"))
for k in order:
    if k not in rows: continue
    d = rows[k]; bb = d.get("by_bucket", {})
    ml = bb.get("multilingual", {}); co = bb.get("constrained", {})
    print("%-12s %6.1f%% %9s   %s   %s" % (
        k, d["loop_pct"], "%d/%d" % (d["loops"], d["n"]),
        "%d/%d" % (ml.get("loops", 0), ml.get("n", 0)),
        "%d/%d" % (co.get("loops", 0), co.get("n", 0))))
a2 = rows.get("a2-pes120", {}).get("loop_pct")
if a2:
    print("\nA2 screen=%.1f%% <-> 3%% full-bench. Target <1%% full-bench => screen <~ %.1f%%." % (a2, a2/3.0))
    for k in screened:
        if k in rows:
            lp = rows[k]["loop_pct"]
            v = "PASS proxy" if lp <= a2/3.0 else ("BELOW A2" if lp < a2 else "no better")
            print("  %-10s %.1f%%  %s" % (k, lp, v))
PY
echo "==================== T176.6 DONE $(date +%H:%M:%S) ===================="
