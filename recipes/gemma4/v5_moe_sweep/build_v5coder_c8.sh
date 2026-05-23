#!/bin/bash
# build_v5coder_c8.sh — user-defined per-layer floor profile (T27 / C8).
#
# Smooth C6-style gradient between user-specified anchors:
#   - L0=95 (unique global peak, "most important")
#   - L0-L4: important descending zone (95, 94, 93, 92, 91)
#   - L5-L9: transition (88..80)
#   - L10-L15: diffuse trough (78, 76, 75, 75, 76, 78)
#   - L16-L24: transition back up + slope toward L25 trough (80..78)
#   - L25=70 (deepest floor, "most diffuse")
#   - L26-L28: ramp back up (78, 85, 90)
#   - L29=95 (high anchor, "important", contra C6 where L29 was lowest)
#
# Total swap slots ~436 (vs C6 ~445). Same recipe shape as C6:
#   STRATEGY=max, WEIGHTS=1 1 3 1 1 2 2 2, PROTECT=16, BREADTH_BONUS=0.5.
# Only the floor map differs.
#
# Hypothesis: stop-signal lives in L29 (contra C6's top98_mean-based
# linear scaling which treated L29 as least important), AND extends
# through L4 in the early-layer zone. L10-L15 plus L25 are the real
# absorbance zones for v5-coder swaps without rumination.
#
# Floor map: scripts/v4_layer_floor_map_c8.json.

set -euo pipefail
cd /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models

TAG="C8_v4floor_perlayer_userprofile_breadth50"
STRATEGY="max"
WEIGHTS="1 1 3 1 1 2 2 2"
PROTECT="16"
V4_FLOOR_MAP="scripts/v4_layer_floor_map_c8.json"
BREADTH_BONUS="0.5"
V5JSON="scripts/expert_neuron_v5_code_fixed.json"
V4JSON="scripts/expert_neuron_v4.json"
SRC_HF="google/gemma-4-26B-A4B-it"
LOGS="logs/v5coder_${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS"

DROP_MAP="scripts/v5coder_${TAG}_drop_map.json"
HF_DIR="google/gemma-4-A4B-98e-v5coder-${TAG}"
NVFP="${HF_DIR}-NVFP4A16"

[ -f "$V5JSON" ] || { echo "[FATAL] $V5JSON not found"; exit 1; }
[ -f "$V4JSON" ] || { echo "[FATAL] $V4JSON not found"; exit 1; }
[ -f "$V4_FLOOR_MAP" ] || { echo "[FATAL] $V4_FLOOR_MAP not found"; exit 1; }
[ -d "$SRC_HF" ] || { echo "[FATAL] $SRC_HF not found"; exit 1; }
echo "[preflight] sources OK"
echo "[preflight] tag=$TAG v4_floor_map=$V4_FLOOR_MAP breadth=$BREADTH_BONUS weights='$WEIGHTS' protect-top=$PROTECT"

# --- Phase 1: drop map ---
if [ -f "$DROP_MAP" ]; then
    echo "[1] drop map exists, skip"
else
    echo "[1] $(date +%H:%M:%S) generate drop map (per-layer floor)"
    /root/anaconda3/envs/omnimergekit/bin/python scripts/generate_drop_map_v5.py \
        --data "$V5JSON" \
        --target 98 \
        --protect-top "$PROTECT" \
        --alpha 2.0 \
        --strategy "$STRATEGY" \
        --normalize rank \
        --class-weights $WEIGHTS \
        --v4-floor-data "$V4JSON" \
        --v4-floor-map "$V4_FLOOR_MAP" \
        --breadth-bonus "$BREADTH_BONUS" \
        --output "$DROP_MAP" \
        2>&1 | tee "$LOGS/dropmap.log"
fi

# --- Phase 2: expert_drop ---
if [ -f "$HF_DIR/model.safetensors.index.json" ]; then
    echo "[2] HF dir exists, skip"
else
    echo "[2] $(date +%H:%M:%S) expert_drop.py"
    /root/anaconda3/envs/omnimergekit/bin/python scripts/expert_drop.py \
        --source-dir "$SRC_HF" \
        --drop-map "$DROP_MAP" \
        --suffix="-v5coder-${TAG}" \
        2>&1 | tee "$LOGS/drop.log"
fi

# --- Phase 3: NVFP4A16 quant ---
if [ -f "$NVFP/hf_quant_config.json" ]; then
    echo "[3] NVFP4A16 exists, skip"
else
    echo "[3] $(date +%H:%M:%S) quantize_any.py --method nvfp4a16"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    /root/anaconda3/envs/modelopt/bin/python /shared/dev/omnimergekit/scripts/quantize_any.py \
        --src "$HF_DIR" --dst "$NVFP" --method nvfp4a16 \
        2>&1 | tee "$LOGS/quant.log"
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
        print("[3] synth preprocessor_config.json from feature_extractor")
except Exception as ex:
    print(f"[3] preprocessor synth skipped: {ex}")
PY
fi

echo
echo "===== v5-coder $TAG build complete ====="
echo "drop_map: $DROP_MAP"
echo "HF dir:   $HF_DIR ($(du -sh $HF_DIR 2>/dev/null | cut -f1))"
echo "NVFP:     $NVFP ($(du -sh $NVFP 2>/dev/null | cut -f1))"
echo "logs:     $LOGS/"
echo
echo "Next: smoke triad gsm8k_30 + humanevalplus_30 + lcb_medium_15 on $NVFP"
