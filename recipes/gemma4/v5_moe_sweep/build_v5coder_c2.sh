#!/bin/bash
# build_v5coder_c2.sh — build v5-coder-C2 with v4-floor + breadth-bonus.
#
# C1 outcome (2026-05-17 smoke): C1_max_codetb scored 18/30 gsm, 19/30 HE+,
# 3/15 LCB. Decisively rejected; LCB failure histogram showed 11/12 fails
# are Python parse errors (rumination format pathology), not algorithmic
# mistakes.
#
# Diagnostic (scripts/diff_v4_vs_c1_drop_maps.py):
#   C1 swapped 503 / 2940 = 17.1% of v4's keep-set. 71% of LOST experts
#   peaked on creative/science/logic in v4 scoring; 94% of GAINED experts
#   peaked on code-axis classes in v5 scoring. Multi-class generalists v4
#   keeps (output-discipline carriers) were systematically demoted by C1's
#   class-weighted aggregation.
#
# C2 design (this build) addresses the root cause:
#   (1) v4-floor protection (--v4-floor-top 80): top-80 experts per layer
#       by v4 pooled-max-of-classes score are KEPT regardless of v5 score.
#       By construction: every v4-top-80 expert survives → C2 inherits the
#       v4 spine that carries output discipline.
#   (2) Multi-class breadth bonus (--breadth-bonus 0.5): aggregate score
#       gets λ·mean(rank_norm) added on top of class-weighted max. Rewards
#       experts that score moderately on many classes (the v3-era insight
#       — multi-class generalists are crucial even when not class-specific).
#
# Same C1 weighting on the remaining 48 (= 128-80) candidate slots:
#   gen-code=3, tgt-HE/HE+/LCB=2 each, others=1.
#
# Expected: 255 swap from v4 (8.7%) vs C1's 503 (17.1%) — C2 keeps
# more of v4's breadth backbone while allowing targeted code-class
# specialists in the marginal zone.

set -euo pipefail
cd /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models

TAG="C2_v4floor80_breadth50"
STRATEGY="max"
WEIGHTS="1 1 3 1 1 2 2 2"
PROTECT="16"
V4_FLOOR_TOP="80"
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
