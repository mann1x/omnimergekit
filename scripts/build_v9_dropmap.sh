#!/usr/bin/env bash
# v9 agentic drop map — the STD16 (published v7-coder) recipe, two defects fixed.
#
# BASIS (verified by hash+diff 2026-09-16, NOT by a source comment):
#   published v7-coder = STD16 = v8coder_fk16_drop_map.json
#   v7coder_C6v3lcb_drop_map.json == v7coder_ml1_drop_map.json (sha 7bfb3591...)
#     = the SAME map BEFORE force-keep; fk16 = it AFTER (46 rescued / 46 evicted).
#
# WHAT CHANGES vs STD16:
#   1. map data: tier_a_max_tokens 128 -> 2048 (at 128 every Tier-A generation was cut
#      mid-stream; measured natural termination is 549-605 tokens), and tier_b 360
#      single-turn zero-tool completions -> 43 multi-turn agentic SWE trajectories, 7 langs.
#   2. force-keep: the shipped 46 pins were built on <|tool_response> (id 50, the PER-STEP
#      stop). Overlap with the <turn|> (id 106, DELIVERY close) lift pins is exactly 0.
#      v7-coder terminates but does not converge -- it protected the wrong terminator.
#
# CLASS WEIGHTS are derived from the map's OWN metadata by v9_classes.py, which hard-fails
# on an unknown category rather than guessing. Two honest deviations from STD16, both
# forced by the new basis and NOT silent:
#   * generic_multilingual is ABSENT (it requires --multilingual-calib, not passed to the
#     map build). STD16 gave it w=0, and under strategy=max a w=0 class can never win the
#     max, so omitting it is equivalent -- though it does change the breadth_bonus
#     denominator, which differs anyway (12 classes here vs 9 in STD16).
#   * ALL targeted_* get w=2. STD16 had 3 targeted benches and zeroed two of them
#     (targeted_humaneval=0, targeted_humanevalplus=0, targeted_lcb_medium_55=2) because
#     that set was heterogeneous: two saturated HumanEval variants plus one discriminating
#     LCB. v9's 7 targeted categories are homogeneous agentic SWE traces, one per language,
#     with no saturated subset to zero -- so every one carries the weight STD16 gave its
#     one discriminating code bench. w is a per-class MULTIPLIER under max, not a mass to
#     divide: splitting 2/7 would drop a language specialist from 2.0 to 0.29.
set -euo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
G=/mnt/sdc/ml/google
DATA=$G/expert_neuron_v9_agentic_code.json
OUT=$G/v9_agentic_drop_map.json
FK=$(cat $G/eog_v4_t106_lift_pins.txt)

mapfile -t CW < <("$PY" /mnt/sdc/harness/v9_classes.py "$DATA")
CLASSES="${CW[0]}"; WEIGHTS="${CW[1]}"
echo "[v9drop] classes: $CLASSES"
echo "[v9drop] weights: $WEIGHTS"

# shellcheck disable=SC2086
"$PY" /srv/ml/scripts/gen_drop_v5_fk.py \
  --data "$DATA" \
  --target 98 --protect-top 16 --alpha 2.0 --score-mode legacy \
  --strategy max --normalize rank \
  --classes $CLASSES \
  --class-weights $WEIGHTS \
  --protect-strategy same --class-protect-floor 0 \
  --v4-floor-data $G/expert_neuron_base_v7.json \
  --v4-floor-top 0 \
  --v4-floor-map /srv/ml/repos/omnimergekit/scripts/v4_layer_floor_map_v7.json \
  --breadth-bonus 0.5 --outlier-wnorm-thresh 10000.0 --outlier-mode median \
  --baseline-drop-map /srv/ml/scripts/teacher_force_98e_p16_clean.json \
  --force-keep "$FK" \
  --output "$OUT"
echo "V9DROP_RC=$?"

echo "[v9drop] === diff vs PUBLISHED v7-coder (STD16 fk16) ==="
"$PY" /srv/ml/repos/omnimergekit/scripts/compare_drop_maps.py \
  --a /srv/ml/repos/omnimergekit/scripts/v8coder_fk16_drop_map.json \
  --b "$OUT" --label-a v7-coder-STD16 --label-b v9-agentic --show-layers 6
echo "V9DROP_CHAIN_DONE"
