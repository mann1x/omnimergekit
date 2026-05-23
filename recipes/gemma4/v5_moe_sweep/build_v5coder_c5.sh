#!/bin/bash
# build_v5coder_c5.sh — v4-floor=95 guardrail variant.
#
# Diagnostic from diff_v4_vs_c2_drop_maps.py (2026-05-17):
#   C2's 255 swaps (v4floor=80) are concentrated at v4-rank 29-47
#   (the marginal zone just above the floor). The top-20 lost experts
#   ALL sit at v4-rank 46-47 — literally 1-2 slots above the protection
#   cliff. Lost class distribution: logic 26%, creative 23%, science 20%
#   — multi-class generalists that v4's max-of-classes scoring valued
#   but C2's class-weighted scoring demoted (their code score is modest).
#
# Hypothesis: raise floor 80 → 95. Only 3 swap slots per layer (≈ 90
# total swaps vs C2's 255). All v4-rank>=33 experts preserved, which
# includes every single top-20 lost expert from C2. Expected outcome:
# tokens close to v4 anchor (76 on gsm, 75 on HE+/30), score within
# ±v5-nudge range, RUMINATION SUPPRESSED.
#
# If C5 token usage ~v4 and score ≈ v4 ± 1pp on each bench: hypothesis
# confirmed, the marginal-zone leak is the rumination cause. Then can
# tune floor between 90-95 for max score gain at safe tokens.

set -euo pipefail
cd /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models

TAG="C5_v4floor95_breadth50"
STRATEGY="max"
WEIGHTS="1 1 3 1 1 2 2 2"
PROTECT="16"
V4_FLOOR_TOP="95"
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
[ -d "$SRC_HF" ] || { echo "[FATAL] $SRC_HF not found"; exit 1; }
echo "[preflight] sources OK"
echo "[preflight] tag=$TAG v4_floor_top=$V4_FLOOR_TOP breadth=$BREADTH_BONUS weights='$WEIGHTS' protect-top=$PROTECT"

# --- Phase 1: drop map ---
if [ -f "$DROP_MAP" ]; then
    echo "[1] drop map exists, skip"
else
    echo "[1] $(date +%H:%M:%S) generate drop map"
    /root/anaconda3/envs/omnimergekit/bin/python scripts/generate_drop_map_v5.py \
        --data "$V5JSON" \
        --target 98 \
        --protect-top "$PROTECT" \
        --alpha 2.0 \
        --strategy "$STRATEGY" \
        --normalize rank \
        --class-weights $WEIGHTS \
        --v4-floor-data "$V4JSON" \
        --v4-floor-top "$V4_FLOOR_TOP" \
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
echo "Next: launch longer-smoke triad on $NVFP via longer_smoke_v5coder.sh"
echo "(remember to update CANDIDATES=(\"$TAG\") in that script first)"
