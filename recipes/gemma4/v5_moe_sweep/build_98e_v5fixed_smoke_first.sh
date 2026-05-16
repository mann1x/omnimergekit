#!/bin/bash
# build_98e_v5fixed_smoke_first.sh
#
# Two-phase A/B builder for v5-fixed-98e (lessons from council
# csl-2026-05-15-1439-4ff0 + user feedback 2026-05-15):
#   Phase A: build + smoke gsm8k_30 on ALL variants (tb10, tb15)
#   Phase B: full triad (HE-164, HE+, LCB-55) on the smoke winner only
#
# NEVER run Phase B before Phase A on all variants. The previous v5h
# pass burned ~2 h on tb10 full HE-164 (which scored 0%) before tb15
# was even built — exactly the failure mode this script exists to prevent.
#
# Input data: scripts/expert_neuron_v5_code_fixed.json
#   (regenerated 2026-05-15 with council bug-fixes:
#     #1 full_seq = prompt+gen at line 210
#     #2 chunk_input overlap=0 default
#     #3 fp16/CUDA precision parity with v4 fp16/CPU reference)
set -euo pipefail
cd /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models

V5JSON="scripts/expert_neuron_v5_code_fixed.json"
SRC_HF="google/gemma-4-26B-A4B-it"
RESULTS="eval_results_vllm_suite/v5fixed"
PORT=8194
LOGS="logs/v5fixed_smoke_first_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS" "$RESULTS"

# --- Preflight ---
if [ ! -f "$V5JSON" ]; then
    echo "[FATAL] $V5JSON not found. v5-fixed regen has not completed."
    exit 1
fi
# Sanity: must have 5 generic + 3 targeted categories
N_CATS=$(python3 -c "import json; d=json.load(open('$V5JSON')); print(len(d['categories']))")
if [ "$N_CATS" != "8" ]; then
    echo "[FATAL] $V5JSON has $N_CATS categories, expected 8 (5 generic + 3 targeted)."
    exit 1
fi
echo "[preflight] $V5JSON has $N_CATS categories — OK"

# --- Variants (name : class-weight tag : tier-b weights) ---
VARIANTS=(
    "tb10:1 1 1 1 1 1.0 1.0 1.0"
    "tb15:1 1 1 1 1 1.5 1.5 1.5"
)

# ───────────── helpers ─────────────
eval_one() {
    local TAG=$1
    local TPL=$2
    local NVFP=$3
    local SERVED="98e_v5fixed_${TAG}_nvfp4a16"
    echo
    echo "----- [$SERVED/$TPL] $(date +%H:%M:%S) start -----"
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
        --results-dir "$RESULTS/$TAG/$TPL" \
        2>&1 | tee -a "$LOGS/${TAG}_${TPL}.log"
}

parse_score() {
    # Args: TAG TPL — prints "strict flex" floats
    local TAG=$1 TPL=$2
    local RES_DIR="$RESULTS/$TAG/$TPL/$TPL/98e_v5fixed_${TAG}_nvfp4a16/lm_eval_out/98e_v5fixed_${TAG}_nvfp4a16"
    local LAST_RES=$(ls -t "$RES_DIR"/results_*.json 2>/dev/null | head -1 || true)
    if [ -z "$LAST_RES" ]; then echo "0 0"; return; fi
    /root/anaconda3/envs/lightseek/bin/python -c "
import json
r = json.load(open('$LAST_RES'))
v = list(r['results'].values())[0]
s = v.get('exact_match,strict-match', v.get('pass@1,extract_chat', v.get('pass@1', 0.0)))
f = v.get('exact_match,flexible-extract', s)
print(f'{s} {f}')
"
}

build_variant() {
    # Builds drop map + expert_drop + NVFP4A16 quant.
    # Prints NVFP path on success. Exits 1 on failure.
    local TAG=$1
    local CW=$2  # space-separated class weights, 8 values
    local DROP_MAP="scripts/v5fixed_${TAG}_98e_p16_drop_map.json"
    local HF_DIR="google/gemma-4-A4B-98e-v5fixed-${TAG}"
    local NVFP="${HF_DIR}-NVFP4A16"

    echo
    echo "===== $(date) v5fixed-${TAG} BUILD start ====="
    echo "  data:     $V5JSON"
    echo "  weights:  $CW"
    echo "  drop map: $DROP_MAP"
    echo "  HF dir:   $HF_DIR"
    echo "  NVFP4A16: $NVFP"

    # 1) drop map (skip if already on disk)
    if [ ! -f "$DROP_MAP" ]; then
        echo "===== [1/4] generate_drop_map_v5.py ====="
        /root/anaconda3/envs/lightseek/bin/python scripts/generate_drop_map_v5.py \
            --data "$V5JSON" \
            --target 98 \
            --protect-top 16 \
            --alpha 2.0 \
            --strategy max \
            --normalize rank \
            --class-weights $CW \
            --output "$DROP_MAP" \
            2>&1 | tee "$LOGS/${TAG}_dropmap.log"
    else
        echo "===== [1/4] $DROP_MAP exists, skip ====="
    fi

    # 2) expert_drop
    if [ ! -f "$HF_DIR/model.safetensors.index.json" ]; then
        echo "===== [2/4] expert_drop.py ====="
        /root/anaconda3/envs/lightseek/bin/python scripts/expert_drop.py \
            --source-dir "$SRC_HF" \
            --drop-map "$DROP_MAP" \
            --suffix="-v5fixed-${TAG}" \
            2>&1 | tee "$LOGS/${TAG}_drop.log"
    else
        echo "===== [2/4] $HF_DIR exists, skip ====="
    fi

    # 3) NVFP4A16 quant
    if [ ! -f "$NVFP/hf_quant_config.json" ]; then
        echo "===== [3/4] quantize_any.py --method nvfp4a16 ====="
        PYTHONDONTWRITEBYTECODE=1 \
        /root/anaconda3/envs/modelopt/bin/python /shared/dev/omnimergekit/scripts/quantize_any.py \
            --src "$HF_DIR" --dst "$NVFP" --method nvfp4a16 \
            2>&1 | tee "$LOGS/${TAG}_quant.log"
    else
        echo "===== [3/4] $NVFP exists, skip ====="
    fi

    # 4) preprocessor synth (Gemma 4 needs this for vLLM 0.20+)
    [ -f "$SRC_HF/processor_config.json" ] && cp -v "$SRC_HF/processor_config.json" "$NVFP/" 2>/dev/null || true
    if [ ! -f "$NVFP/preprocessor_config.json" ]; then
        /root/anaconda3/envs/modelopt/bin/python - <<PY
import json, pathlib
pc = json.load(open(pathlib.Path("$SRC_HF") / "processor_config.json"))
fe = pc.get("feature_extractor", {})
if fe:
    json.dump(fe, open(pathlib.Path("$NVFP") / "preprocessor_config.json", "w"), indent=2)
    print("synth preprocessor_config.json from feature_extractor")
PY
    fi

    echo "$NVFP" > "$LOGS/${TAG}_nvfp_path.txt"
}

# ───────────── PHASE A — build + smoke all variants ─────────────
echo
echo "===== PHASE A: build + smoke gsm8k_30 on ALL variants ====="
declare -A SCORES_STRICT SCORES_FLEX NVFP_PATHS
for V in "${VARIANTS[@]}"; do
    TAG=${V%%:*}
    CW=${V#*:}
    build_variant "$TAG" "$CW"
    NVFP=$(cat "$LOGS/${TAG}_nvfp_path.txt")
    NVFP_PATHS[$TAG]="$NVFP"

    echo "===== [4/4] smoke gsm8k_30 on $TAG ====="
    eval_one "$TAG" "gsm8k_30" "$NVFP"

    read STRICT FLEX < <(parse_score "$TAG" "gsm8k_30")
    SCORES_STRICT[$TAG]=$STRICT
    SCORES_FLEX[$TAG]=$FLEX
    echo "[smoke] $TAG gsm8k_30 strict=${STRICT} flex=${FLEX}"
done

# ───────────── decide winner ─────────────
echo
echo "===== PHASE A summary ====="
WINNER=""
WINNER_FLEX=0
for TAG in "${!SCORES_FLEX[@]}"; do
    echo "  $TAG: strict=${SCORES_STRICT[$TAG]} flex=${SCORES_FLEX[$TAG]}"
    # Compare flex (float) — winner = highest flex; tie-break = strict
    cmp=$(/root/anaconda3/envs/lightseek/bin/python -c "
a = float('${SCORES_FLEX[$TAG]}'); b = float('${WINNER_FLEX:-0}')
print('gt' if a > b else ('eq' if a == b else 'lt'))
")
    if [ "$cmp" = "gt" ]; then
        WINNER=$TAG
        WINNER_FLEX=${SCORES_FLEX[$TAG]}
    fi
done

if [ -z "$WINNER" ]; then
    echo "[FATAL] no winner could be selected"; exit 2
fi

# Gate: winner flex must be >= 60% (math intact), else both variants
# collapsed math and full triad is pointless.
GATE_OK=$(/root/anaconda3/envs/lightseek/bin/python -c "print('1' if float('$WINNER_FLEX') >= 0.60 else '0')")
if [ "$GATE_OK" != "1" ]; then
    echo "[ABORT] winner $WINNER flex=${WINNER_FLEX} < 0.60 — math collapsed on all variants."
    echo "         Investigate map quality before running full triad."
    exit 3
fi

echo
echo "===== WINNER: $WINNER (flex=${WINNER_FLEX}) ====="

# ───────────── PHASE B — full triad on winner only ─────────────
echo
echo "===== PHASE B: full triad on $WINNER ====="
NVFP="${NVFP_PATHS[$WINNER]}"
for TPL in humaneval_full humanevalplus_full lcb_medium_55_v4; do
    eval_one "$WINNER" "$TPL" "$NVFP"
done

echo
echo "===== $(date) v5fixed smoke-first DONE — winner=$WINNER ====="
echo "Note: the OTHER variant is built and quantized; if you want to"
echo "      run its full triad later, point eval_one at NVFP_PATHS[<tag>]."
for TAG in "${!NVFP_PATHS[@]}"; do
    [ "$TAG" = "$WINNER" ] && continue
    echo "  $TAG NVFP: ${NVFP_PATHS[$TAG]}"
done
