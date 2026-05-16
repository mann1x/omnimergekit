#!/bin/bash
# build_98e_v5fixed_sweep.sh — 13-variant aggregator sweep for 98e-v5-fixed.
#
# Designed per user direction 2026-05-16: the v5-fixed JSON is sound;
# the broken thing is the choice of `--strategy max --normalize rank
# --class-weights uniform` in the published tb10 baseline (gsm8k_30 = 50%).
# Probe the aggregation space WITHOUT retreating to v4's pooled-single-class
# methodology, which had a lower ceiling.
#
# Holds: normalize=rank (per user — they like rank's heavy-tail safety)
# Sweeps:
#   Group A (uniform weights, p16, same protect-strategy): 6 strategies
#       lp4 / top3mean / softmax_t4 / second / mean / geomean
#       (A1=max already done as v5fixed-tb10, skip)
#   Group B (max strategy, p16, same protect-strategy): 4 weight patterns
#       genheavy / mathcode / tgtheavy / broad
#   Group C (max strategy, uniform, p16): 3 protect-axis variants
#       protect_top=20 / class_protect_floor=2 / protect_strategy=sum
#
# Per variant: generate map → expert_drop → NVFP4A16 quant → smoke triad
#   (gsm8k_30 + humaneval_1_smoke + lcb_medium_1_smoke ~30 min/variant)
# Total wall: ~11h on a single 3090 (build CPU + quant GPU + smoke GPU).
# Disk: ~52 GB/variant × 13 = ~680 GB (fits in 6.5 TB free).
#
# Idempotent: skips any phase whose output artifact already exists. Resumes
# cleanly across crashes; just re-run the script.
#
# At end: prints summary table and per-variant smoke scores; PROMOTES the
# top-3 (by composite score) to a Phase-B full-triad run on the SAME smoke
# helper (called separately so the user can review the table first).

set -euo pipefail
cd /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models

V5JSON="scripts/expert_neuron_v5_code_fixed.json"
SRC_HF="google/gemma-4-26B-A4B-it"
RESULTS="eval_results_vllm_suite/v5fixed_sweep"
PORT=8194
LOGS="logs/v5fixed_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS" "$RESULTS"

# --- Preflight ---
if [ ! -f "$V5JSON" ]; then
    echo "[FATAL] $V5JSON not found"; exit 1
fi
N_CATS=$(python3 -c "import json; d=json.load(open('$V5JSON')); print(len(d['categories']))")
[ "$N_CATS" != "8" ] && { echo "[FATAL] $V5JSON has $N_CATS categories, expected 8"; exit 1; }
echo "[preflight] $V5JSON has 8 categories — OK"

# Verify the patched aggregator strategies are present
PATCHED_OK=$(/root/anaconda3/envs/lightseek/bin/python -c "
import subprocess
r = subprocess.run(['/root/anaconda3/envs/lightseek/bin/python',
                    'scripts/generate_drop_map_v5.py', '--help'],
                    capture_output=True, text=True)
needed = ['lp4', 'softmax_t4', 'second', 'top3mean']
print('YES' if all(s in r.stdout for s in needed) else 'NO')
")
if [ "$PATCHED_OK" != "YES" ]; then
    echo "[FATAL] generate_drop_map_v5.py missing new strategies"; exit 1
fi
echo "[preflight] aggregator strategies (lp4, top3mean, softmax_t4, second) present — OK"

# --- Variants ---
# Schema: tag:strategy:weights(8 space-separated):protect_top:floor:protect_strategy
#
# Pre-flight overlap analysis 2026-05-16 found that the original Group C
# (protect-axis: protect_top, class_protect_floor, protect_strategy) is
# DEGENERATE on the v5-fixed data — all variants produce identical drop maps
# (30/30 overlap with the no-modification baseline). Mechanism: rank-normalized
# scores make "low-max-aggregate" and "low-sum-aggregate" the same set of
# experts on this distribution. Replacing Group C with weight × strategy
# crossovers (Group D), which empirically show meaningful divergence.
VARIANTS=(
    # Group A — strategy axis, uniform weights, protect_top=16
    "A2_lp4_uni:lp4:1 1 1 1 1 1 1 1:16:0:same"
    "A3_top3mean_uni:top3mean:1 1 1 1 1 1 1 1:16:0:same"
    "A4_softmax_t4_uni:softmax_t4:1 1 1 1 1 1 1 1:16:0:same"
    "A5_second_uni:second:1 1 1 1 1 1 1 1:16:0:same"

    # Group B — weight axis, strategy=max, protect_top=16
    "B1_max_genheavy:max:2 2 2 2 2 1 1 1:16:0:same"
    "B2_max_mathcode:max:3 1 2 1 1 1 1 1:16:0:same"
    "B3_max_tgtheavy:max:1 1 1 1 1 3 3 3:16:0:same"
    "B4_max_broad:max:2 1 3 2 1 2 2 2:16:0:same"
    "B5_max_xtgt:max:1 1 1 1 1 5 5 5:16:0:same"

    # Group D — strategy × weight crossovers (interaction effects)
    "D1_lp4_mathcode:lp4:3 1 2 1 1 1 1 1:16:0:same"
    "D2_second_mathcode:second:3 1 2 1 1 1 1 1:16:0:same"
    "D3_top3mean_genheavy:top3mean:2 2 2 2 2 1 1 1:16:0:same"
    "D4_second_tgtheavy:second:1 1 1 1 1 3 3 3:16:0:same"
)
echo "[sweep] ${#VARIANTS[@]} variants queued; smoke = gsm8k_30 + humaneval_1_smoke + lcb_medium_1_smoke"

# --- helpers ---
declare -A V_GSM_FLEX V_HE_PASS V_LCB_PASS

build_drop_map() {
    local TAG=$1 STRATEGY=$2 WEIGHTS=$3 PROTECT=$4 FLOOR=$5 PSTRAT=$6
    local DROP_MAP="scripts/v5fixed_sweep_${TAG}_drop_map.json"
    if [ -f "$DROP_MAP" ]; then
        echo "[$TAG/1] drop map exists, skip"; return
    fi
    echo "[$TAG/1] generate drop map (strategy=$STRATEGY weights='$WEIGHTS' p=$PROTECT floor=$FLOOR pstrat=$PSTRAT)"
    local FLOOR_ARG=""
    [ "$FLOOR" != "0" ] && FLOOR_ARG="--class-protect-floor $FLOOR"
    local PSTRAT_ARG=""
    [ "$PSTRAT" != "same" ] && PSTRAT_ARG="--protect-strategy $PSTRAT"
    /root/anaconda3/envs/lightseek/bin/python scripts/generate_drop_map_v5.py \
        --data "$V5JSON" \
        --target 98 \
        --protect-top "$PROTECT" \
        --alpha 2.0 \
        --strategy "$STRATEGY" \
        --normalize rank \
        --class-weights $WEIGHTS \
        $FLOOR_ARG $PSTRAT_ARG \
        --output "$DROP_MAP" \
        2>&1 | tee "$LOGS/${TAG}_dropmap.log"
}

build_hf_dir() {
    local TAG=$1
    local DROP_MAP="scripts/v5fixed_sweep_${TAG}_drop_map.json"
    local HF_DIR="google/gemma-4-A4B-98e-v5fixed-sweep-${TAG}"
    if [ -f "$HF_DIR/model.safetensors.index.json" ]; then
        echo "[$TAG/2] HF dir exists, skip"; return
    fi
    echo "[$TAG/2] expert_drop.py"
    /root/anaconda3/envs/lightseek/bin/python scripts/expert_drop.py \
        --source-dir "$SRC_HF" \
        --drop-map "$DROP_MAP" \
        --suffix="-v5fixed-sweep-${TAG}" \
        2>&1 | tee "$LOGS/${TAG}_drop.log"
}

build_nvfp() {
    local TAG=$1
    local HF_DIR="google/gemma-4-A4B-98e-v5fixed-sweep-${TAG}"
    local NVFP="${HF_DIR}-NVFP4A16"
    if [ -f "$NVFP/hf_quant_config.json" ]; then
        echo "[$TAG/3] NVFP4A16 exists, skip"
    else
        echo "[$TAG/3] quantize_any.py --method nvfp4a16"
        PYTHONDONTWRITEBYTECODE=1 \
        /root/anaconda3/envs/modelopt/bin/python /shared/dev/omnimergekit/scripts/quantize_any.py \
            --src "$HF_DIR" --dst "$NVFP" --method nvfp4a16 \
            2>&1 | tee "$LOGS/${TAG}_quant.log"
    fi
    # preprocessor synth (Gemma 4 vLLM 0.20+ needs it)
    [ -f "$SRC_HF/processor_config.json" ] && cp -v "$SRC_HF/processor_config.json" "$NVFP/" 2>/dev/null || true
    if [ ! -f "$NVFP/preprocessor_config.json" ]; then
        /root/anaconda3/envs/modelopt/bin/python - <<PY
import json, pathlib
try:
    pc = json.load(open(pathlib.Path("$SRC_HF") / "processor_config.json"))
    fe = pc.get("feature_extractor", {})
    if fe:
        json.dump(fe, open(pathlib.Path("$NVFP") / "preprocessor_config.json", "w"), indent=2)
        print("[$TAG/3] synth preprocessor_config.json from feature_extractor")
except Exception as ex:
    print(f"[$TAG/3] preprocessor synth skipped: {ex}")
PY
    fi
    echo "$NVFP"
}

eval_one() {
    local TAG=$1 TPL=$2 NVFP=$3
    local SERVED="98e_v5fixed_sweep_${TAG}_nvfp4a16"
    local OUTDIR="$RESULTS/$TAG/$TPL"
    # Idempotency: skip if a results_*.json already exists for this template
    if ls "$OUTDIR/$TPL/$SERVED/lm_eval_out/$SERVED"/results_*.json >/dev/null 2>&1 || \
       ls "$OUTDIR/$TPL/$SERVED"/results_*.json >/dev/null 2>&1; then
        echo "[$TAG/$TPL] results exist, skip"; return
    fi
    echo "[$TAG/$TPL] $(date +%H:%M:%S) start"
    PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH PYTHONDONTWRITEBYTECODE=1 \
    VLLM_PYTHON=/root/anaconda3/envs/vllm/bin/python \
    /root/anaconda3/envs/omnimergekit/bin/python /shared/dev/omnimergekit/eval/omk_eval.py \
        --model "$NVFP" \
        --template "$TPL" \
        --backend vllm \
        --port "$PORT" \
        --served-name "$SERVED" \
        --tokenizer "$SRC_HF" \
        --max-model-len 40960 \
        --results-dir "$OUTDIR" \
        2>&1 | tee -a "$LOGS/${TAG}_${TPL}.log"
}

parse_gsm_score() {
    local TAG=$1
    local SERVED="98e_v5fixed_sweep_${TAG}_nvfp4a16"
    local R_DIR="$RESULTS/$TAG/gsm8k_30/gsm8k_30/$SERVED/lm_eval_out/$SERVED"
    local R=$(ls -t "$R_DIR"/results_*.json 2>/dev/null | head -1 || true)
    [ -z "$R" ] && { echo "0 0"; return; }
    /root/anaconda3/envs/lightseek/bin/python -c "
import json
r=json.load(open('$R'))['results']
v=list(r.values())[0]
s=v.get('exact_match,strict-match',0.0); f=v.get('exact_match,flexible-extract',s)
print(f'{s} {f}')
"
}

parse_he_score() {
    local TAG=$1 TPL=$2
    local SERVED="98e_v5fixed_sweep_${TAG}_nvfp4a16"
    local R_DIR="$RESULTS/$TAG/$TPL/$TPL/$SERVED/lm_eval_out/$SERVED"
    local R=$(ls -t "$R_DIR"/results_*.json 2>/dev/null | head -1 || true)
    [ -z "$R" ] && { echo "0"; return; }
    /root/anaconda3/envs/lightseek/bin/python -c "
import json
r=json.load(open('$R'))['results']
v=list(r.values())[0]
# 1-q smokes report pass@1 with various key names
for k in ('pass@1,extract_chat','pass@1','exact_match,strict-match','exact_match,flexible-extract'):
    if k in v: print(v[k]); break
else: print(0)
"
}

# --- main loop ---
echo
echo "===== $(date) sweep start ====="
SUMMARY="$LOGS/_sweep_summary.tsv"
echo -e "tag\tstrategy\tweights\tprotect\tfloor\tpstrat\tgsm_strict\tgsm_flex\the_smoke\tlcb_smoke" > "$SUMMARY"

for V in "${VARIANTS[@]}"; do
    IFS=':' read -r TAG STRATEGY WEIGHTS PROTECT FLOOR PSTRAT <<< "$V"
    echo
    echo "===================="
    echo "===== $TAG ($STRATEGY, w='$WEIGHTS', p=$PROTECT, floor=$FLOOR, pstrat=$PSTRAT) ====="
    echo "===================="
    build_drop_map "$TAG" "$STRATEGY" "$WEIGHTS" "$PROTECT" "$FLOOR" "$PSTRAT"
    build_hf_dir "$TAG"
    NVFP=$(build_nvfp "$TAG" | tail -1)
    # smoke triad
    for TPL in gsm8k_30 humaneval_1_smoke lcb_medium_1_smoke; do
        eval_one "$TAG" "$TPL" "$NVFP" || echo "[$TAG/$TPL] FAILED"
    done
    # parse and record
    read GSM_S GSM_F < <(parse_gsm_score "$TAG")
    HE=$(parse_he_score "$TAG" humaneval_1_smoke)
    LCB=$(parse_he_score "$TAG" lcb_medium_1_smoke)
    V_GSM_FLEX[$TAG]=$GSM_F
    V_HE_PASS[$TAG]=$HE
    V_LCB_PASS[$TAG]=$LCB
    echo -e "$TAG\t$STRATEGY\t$WEIGHTS\t$PROTECT\t$FLOOR\t$PSTRAT\t$GSM_S\t$GSM_F\t$HE\t$LCB" >> "$SUMMARY"
    echo "[$TAG] gsm_strict=$GSM_S gsm_flex=$GSM_F he_smoke=$HE lcb_smoke=$LCB"
done

echo
echo "===== $(date) sweep DONE ====="
echo
echo "===== summary ====="
column -t -s $'\t' "$SUMMARY"
echo
echo "Summary saved at: $SUMMARY"
echo
echo "Top-3 by composite (gsm_flex + 0.3*he_smoke + 0.3*lcb_smoke):"
sort -t$'\t' -k8 -n -r "$SUMMARY" | head -4
echo
echo "Next: review the table, then run the full triad on top-3 manually."
