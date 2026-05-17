#!/bin/bash
# build_v5coder_c1.sh — build the C1_max_codetb v5-coder candidate.
#
# Rationale (T19.5 longer-smoke + v4 anchor):
#   B2_max_mathcode (weights "3 1 2 1 1 1 1 1" = math 3× + code 2×) hit
#   v4 parity on gsm but lost -55pp HE-20 and -60pp LCB-5 vs v4. The dual
#   weighting strategy (primary 3× + secondary 2×) worked for math; the
#   sweep showed it didn't help code because the secondary class was code
#   *for math*, not the other way around.
#
#   C1 applies the same dual-weighting strategy WITH CODE AS PRIMARY:
#     generic_code 3× (primary, mirrors B2's math 3×)
#     targeted_he / targeted_hep / targeted_lcb 2× (secondary support from
#       Tier-B PASS-trace code observations)
#     everything else 1×
#   = class-weights "1 1 3 1 1 2 2 2"
#
# Hypothesis: this preserves a strong code-class core PLUS the
# observed-correct-code experts as second-order support, instead of
# B3_max_tgtheavy's exclusive Tier-B emphasis which collapsed
# (HE-20=0.20, LCB-5=0.0, silent-empty at p10).
#
# Same recipe as the sweep otherwise: max aggregation, rank normalize,
# protect-top=16, no protect-strategy override.

set -euo pipefail
cd /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models

TAG="C1_max_codetb"
STRATEGY="max"
WEIGHTS="1 1 3 1 1 2 2 2"
PROTECT="16"
V5JSON="scripts/expert_neuron_v5_code_fixed.json"
SRC_HF="google/gemma-4-26B-A4B-it"
LOGS="logs/v5coder_${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS"

DROP_MAP="scripts/v5coder_${TAG}_drop_map.json"
HF_DIR="google/gemma-4-A4B-98e-v5coder-${TAG}"
NVFP="${HF_DIR}-NVFP4A16"

[ -f "$V5JSON" ] || { echo "[FATAL] $V5JSON not found"; exit 1; }
[ -d "$SRC_HF" ] || { echo "[FATAL] $SRC_HF not found"; exit 1; }
echo "[preflight] sources OK"
echo "[preflight] tag=$TAG strategy=$STRATEGY weights='$WEIGHTS' protect-top=$PROTECT"

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
