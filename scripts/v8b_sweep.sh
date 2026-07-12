#!/usr/bin/env bash
# v8b_sweep.sh — ZERO-GPU drop-map sweep for the v8b science<->code trade.
# 1) Reproduce fkbroad EXACTLY (single-variable guard; keep-set must match).
# 2) Sweep candidates varying ONLY science/ml class-weights and the force-keep set.
# 3) Diff each vs fkbroad with v8b_pick.py (loop-safety-aware): sci_in / risky_in /
#    code_out / pins_freed. v8b-safe keepmeta passed as the proven-frontier reference.
# No GPU, no model builds — just drop-map JSONs (8K each).
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
GEN=/srv/ml/scripts/generate_drop_map_v5fk.py
DATA=/mnt/sdc/ml/google/expert_neuron_v7_code.json
AGMAP=/mnt/sdc/ml/google/expert_neuron_v7_agentic_eog_t106.json
FLOOR_DATA=/mnt/sdc/ml/google/expert_neuron_base_v7.json
FLOOR_MAP=/srv/ml/repos/omnimergekit/scripts/v4_layer_floor_map_v7.json
BASELINE=/srv/ml/scripts/teacher_force_98e_p16_clean.json
FKBROAD=/srv/ml/scripts/v8coder_fkbroad_drop_map.json
SAFEKEEP=/mnt/sdc/ml/sft_heal/v8b_safe_keepmeta.json
PICK=/mnt/sdc/ml/gpqa_dissect/v8b_pick.py
OUT=/mnt/sdc/ml/gpqa_dissect/v8b_sweep
PINS="0:2,0:9,0:10,0:86,0:93,1:44,1:92,1:95,2:4,2:113,7:66,8:80,8:114,9:110,10:12,11:8,11:82,11:118,12:6,12:38,13:82,14:45,14:102,15:39,15:47,15:99,17:57,17:71,21:10,23:96"
# pin trims (decided from science_headroom): LCB-strong trio vs generic-code-strong trio
UNPIN_LCB="14:45,21:10,11:118"          # weak generic-code, STRONG LCB
UNPIN_GC="14:102,9:110,15:47"           # STRONG generic-code, weak LCB
CLASSES="generic_math generic_logic generic_code generic_science generic_creative generic_multilingual targeted_humaneval targeted_humanevalplus targeted_lcb_medium_55"
ts(){ date '+%T %Z'; }
mkdir -p "$OUT"
echo "==================== v8b zero-GPU sweep $(ts) ===================="
for f in "$GEN" "$DATA" "$AGMAP" "$FLOOR_DATA" "$FLOOR_MAP" "$BASELINE" "$FKBROAD" "$SAFEKEEP" "$PICK"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done

# drop the un-pinned pins from a CSV pin list -> echo the trimmed CSV
trim_pins(){ # full_csv  remove_csv
  "$PY" - "$1" "$2" <<'EOF'
import sys
full=set(sys.argv[1].split(",")); rm=set(sys.argv[2].split(","))
print(",".join(sorted(full-rm, key=lambda t:(int(t.split(":")[0]),int(t.split(":")[1])))))
EOF
}

gen(){ # label  weights  force_keep_csv
  local label=$1 weights=$2 fk=$3 out="$OUT/v8b_${1}_drop_map.json"
  echo "[$label $(ts)] gen weights=[$weights] pins=$(echo "$fk"|tr ',' ' '|wc -w)"
  "$PY" "$GEN" \
    --data "$DATA" --target 98 --protect-top 16 --alpha 2.0 \
    --score-mode legacy --strategy max --normalize rank \
    --classes $CLASSES \
    --class-weights $weights \
    --protect-strategy same --class-protect-floor 0 \
    --v4-floor-data "$FLOOR_DATA" --v4-floor-top 0 --v4-floor-map "$FLOOR_MAP" \
    --breadth-bonus 0.5 --outlier-wnorm-thresh 10000.0 --outlier-mode median \
    --baseline-drop-map "$BASELINE" \
    --force-keep "$fk" \
    --output "$out" > "$OUT/${label}.gen.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && [ -f "$out" ] || { echo "  FATAL gen rc=$rc (see $OUT/${label}.gen.log)"; tail -5 "$OUT/${label}.gen.log"; return 1; }
}

# --- C0: fkbroad reproduction (guard) ---
gen fkrepro "1 1 3 1 1 0 0 0 2" "$PINS" || exit 3
C0="$OUT/v8b_fkrepro_drop_map.json"
echo "[guard $(ts)] keep-set compare C0 vs fkbroad:"
"$PY" - "$C0" "$FKBROAD" <<'EOF'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
diff=0
for L in map(str,range(30)):
    if sorted(int(x) for x in a[L])!=sorted(int(x) for x in b[L]): diff+=1
print(f"  layers differing: {diff}/30  -> {'IDENTICAL (guard PASS)' if diff==0 else 'MISMATCH (guard FAIL)'}")
sys.exit(0 if diff==0 else 9)
EOF
[ $? -eq 0 ] || { echo "GUARD FAILED — harness does not reproduce fkbroad; STOP."; exit 4; }

# --- sweep candidates ---
gen sci2ml1 "1 1 3 2 1 1 0 0 2" "$PINS"        || exit 5
gen sci3ml2 "1 1 3 3 1 2 0 0 2" "$PINS"        || exit 5
TRIM_LCB=$(trim_pins "$PINS" "$UNPIN_LCB")
TRIM_GC=$(trim_pins "$PINS" "$UNPIN_GC")
gen sci3ml2_unpinLCB "1 1 3 3 1 2 0 0 2" "$TRIM_LCB" || exit 5
gen sci3ml2_unpinGC  "1 1 3 3 1 2 0 0 2" "$TRIM_GC"  || exit 5

echo "==================== v8b SWEEP DIFF (vs fkbroad) $(ts) ===================="
"$PY" "$PICK" "$DATA" "$AGMAP" "$FKBROAD" "$PINS" 16 \
  "C0_fkrepro=$OUT/v8b_fkrepro_drop_map.json" \
  "C1_sci2ml1=$OUT/v8b_sci2ml1_drop_map.json" \
  "C2_sci3ml2=$OUT/v8b_sci3ml2_drop_map.json" \
  "C3_sci3ml2_unpinLCB=$OUT/v8b_sci3ml2_unpinLCB_drop_map.json" \
  "C4_sci3ml2_unpinGC=$OUT/v8b_sci3ml2_unpinGC_drop_map.json" \
  "REF_v8b_safe=$SAFEKEEP"
echo "==================== v8b SWEEP DONE $(ts) ===================="
