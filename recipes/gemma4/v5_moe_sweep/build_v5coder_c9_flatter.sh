#!/bin/bash
# build_v5coder_c6.sh — layer-relevance-weighted v4 floor.
#
# Per-layer floor is scaled by v4 top98_mean per layer:
#   - L0 (top98_mean=514, most active): floor=95
#   - L29 (top98_mean=176, least active): floor=75
#   - Linear in between, avg ≈ 83
#
# Total swap slots ≈ 445 (vs C2=540, C5=90). Hypothesis: layers with
# concentrated experts (low gini, high top98_mean) carry the v4 routing
# discipline; layers with diffuse experts (late layers L25-L29) have
# slack and can absorb v5 swaps without rumination.
#
# Floor map: scripts/v4_layer_floor_map.json (generated 2026-05-17).

set -euo pipefail
cd /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models

TAG="C9flatter_v4floor_perlayer_L0_90_L29_80"
STRATEGY="max"
WEIGHTS="1 1 3 1 1 2 2 2"
PROTECT="16"
V4_FLOOR_MAP="scripts/v4_layer_floor_map_c9_flatter.json"
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
