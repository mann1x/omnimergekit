#!/bin/bash
# pod_quant_31b.sh — build NVFP4A16 quants for Gemma 4 31B base + he1-it on pod.
#
# CANONICAL LOCATION: omnimergekit/scripts/pod_quant_31b.sh
# Mirrors the recipe that built the 128e + v4 NVFP4A16 quants (canonical
# omnimergekit/scripts/quantize_any.py --method nvfp4a16).
#
# Pre-reqs:
#   - /workspace/miniconda/envs/modelopt   (built by pod_setup_eval_envs.sh
#                                           OR earlier bootstrap; contains
#                                           modelopt commit g7a11fb240 + transformers 5.8.x)
#   - /workspace/models/gemma-4-31B-it/                  (BF16 source — google base)
#   - /workspace/models/gemma-4-31b-he1-it/              (BF16 source — ManniX-ITA)
#   - HF_TOKEN exported (for the push step)
#
# Outputs (PRIVATE on HF until eval validates):
#   - /workspace/models/Gemma-4-31B-it-NVFP4A16/
#   - /workspace/models/gemma-4-31b-he1-it-NVFP4A16/
#   - ManniX-ITA/Gemma-4-31B-it-NVFP4A16          (HF, private)
#   - ManniX-ITA/gemma-4-31b-he1-it-NVFP4A16      (HF, private)
#
# Runtime estimate (RTX 3090, sequential): ~6-10 min per model = ~12-20 min total.
# Disk peak during run: ~60 GB per model (BF16 source + working buffers + quant out).
set -euo pipefail

OMK=/workspace/omnimergekit
MODELS=/workspace/models
LOGS=/workspace/logs
mkdir -p "$LOGS"

CALIB_SAMPLES=512  # matches 128e/v4 recipe — activates enough MoE experts (n/a for dense 31B
                   # but kept symmetric for the calibration distribution)

source /workspace/miniconda/etc/profile.d/conda.sh
conda activate modelopt
export HF_XET_HIGH_PERFORMANCE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_TOKEN="${HF_TOKEN:-***REMOVED-DEAD-HF-TOKEN***}"

quant_one() {
    local NAME="$1" SRC="$2" DST="$3" REPO="$4"
    local LOG="$LOGS/quant_${NAME}_$(date +%Y%m%d_%H%M%S).log"
    echo "[$(date)] === quantize $NAME → $DST ==="
    echo "    src=$SRC"
    echo "    repo=$REPO (PRIVATE)"
    echo "    log=$LOG"

    if [ -f "$DST/hf_quant_config.json" ]; then
        echo "    [$NAME] DST already has hf_quant_config.json — skipping quant step"
    else
        python "$OMK/scripts/quantize_any.py" \
            --method nvfp4a16 \
            --src "$SRC" \
            --dst "$DST" \
            --calib-samples "$CALIB_SAMPLES" \
            2>&1 | tee -a "$LOG"
    fi

    echo "[$(date)] === push $REPO (PRIVATE) ==="
    # hf create handles the "already exists" case via --exist-ok
    hf repo create "$REPO" --type model --private --exist-ok 2>&1 | tee -a "$LOG" || true
    hf upload "$REPO" "$DST" . 2>&1 | tail -20 | tee -a "$LOG"

    echo "[$(date)] === $NAME done ==="
}

quant_one  31b      "$MODELS/gemma-4-31B-it"        "$MODELS/Gemma-4-31B-it-NVFP4A16"        "ManniX-ITA/Gemma-4-31B-it-NVFP4A16"
quant_one  31b_he1  "$MODELS/gemma-4-31b-he1-it"    "$MODELS/gemma-4-31b-he1-it-NVFP4A16"    "ManniX-ITA/gemma-4-31b-he1-it-NVFP4A16"

# Drop BF16 sources once quants exist + uploaded — keep disk lean for eval phase.
for SRC in "$MODELS/gemma-4-31B-it" "$MODELS/gemma-4-31b-he1-it"; do
    DST="${SRC%/}-NVFP4A16"
    DST_ALT=$(echo "$DST" | sed 's|gemma-4-31B-it|Gemma-4-31B-it|')
    if [ -f "$DST/hf_quant_config.json" ] || [ -f "$DST_ALT/hf_quant_config.json" ]; then
        echo "[$(date)] purge BF16 src: $SRC"
        rm -rf "$SRC"
    fi
done

df -h /workspace | tail -1
du -sh "$MODELS"/* | sort -h
echo "[$(date)] === ALL QUANTS DONE ==="
