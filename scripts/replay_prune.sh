#!/usr/bin/env bash
# Replay a previously-published head/FFN prune on a fresh host.
#
# Replays the exact prune that produced a published variant by reusing the
# cached importance scores + canary baseline from the original run. Phase 1
# (nf4 importance, ~30 min on 3090) and Phase 0 canary capture (~17 min) are
# skipped — only Phase 0 calib forward + Phase 2 lstsq heal + Phase 2.5
# canary TF-eval still run.
#
# Designed for portable re-runs: solidPC → cloud pod, or one pod → another.
# Uploads the resulting BF16 safetensors to HF when canary passes.
#
# Required cache artifacts (provided as a single source dir, either local
# path or rsync target like 'user@host:/path/to/cache_dir'):
#   gemma4_31b_imp_full_nf4.pt          — Phase 1 importance (26 KB)
#   gemma4_31b_canary_baseline_n50.pt   — Phase 0 canary baseline (5 KB)
#   prune_manifest.json                 — head selection from original run (14 KB)
#                                          (optional but recommended — used for
#                                           bit-identical-selection sanity check)
#
# Usage:
#   bash replay_prune.sh \
#       <base_hf_id> <target_hf_id> <prune_frac> <cache_src> <hf_token> [extra prune_local_heal.py args...]
#
# e.g.
#   bash replay_prune.sh \
#       google/gemma-4-31B-it \
#       ManniX-ITA/gemma-4-31b-he1-it \
#       0.125 \
#       solidpc:/srv/.../backup_models/4b_phase1 \
#       "$(cat ~/.cache/huggingface/token)"
#
# Optional env:
#   WORKDIR              — staging dir, default /workspace/replay_prune
#   RECIPE               — path to prune_local_heal.py, default
#                          /shared/dev/omnimergekit/recipes/gemma4_31b/prune_local_heal.py
#                          (falls back to /workspace/omnimergekit/recipes/... on pod)
#   GPU_MEM, CPU_MEM     — accelerate offload budget, default 8GiB / 200GiB
#   CHUNK_TOKENS         — Phase 0 chunk size, default 384
#   RIDGE                — lstsq ridge, default 1e-2
#   SKIP_UPLOAD=1        — keep weights local, don't push to HF

set -euo pipefail

BASE_HF_ID="${1:?usage: $0 <base_hf_id> <target_hf_id> <prune_frac> <cache_src> <hf_token> [extra...]}"
TARGET_HF_ID="${2:?usage: $0 <base_hf_id> <target_hf_id> <prune_frac> <cache_src> <hf_token> [extra...]}"
PRUNE_FRAC="${3:?usage: $0 <base_hf_id> <target_hf_id> <prune_frac> <cache_src> <hf_token> [extra...]}"
CACHE_SRC="${4:?usage: $0 <base_hf_id> <target_hf_id> <prune_frac> <cache_src> <hf_token> [extra...]}"
HF_TOKEN_ARG="${5:?usage: $0 <base_hf_id> <target_hf_id> <prune_frac> <cache_src> <hf_token> [extra...]}"
shift 5
EXTRA_ARGS=("$@")

if [[ "$HF_TOKEN_ARG" != "-" && -n "$HF_TOKEN_ARG" ]]; then
    export HF_TOKEN="$HF_TOKEN_ARG"
fi
: "${HF_TOKEN:?HF_TOKEN required as arg-5 or env}"

: "${WORKDIR:=/workspace/replay_prune}"
: "${GPU_MEM:=8GiB}"
: "${CPU_MEM:=200GiB}"
: "${CHUNK_TOKENS:=384}"
: "${RIDGE:=1e-2}"

# Locate the recipe — try canonical clone path on pods first, then fall back
# to the dev workspace (solidPC).
if [[ -n "${RECIPE:-}" ]]; then
    :
elif [[ -f /workspace/omnimergekit/recipes/gemma4_31b/prune_local_heal.py ]]; then
    RECIPE=/workspace/omnimergekit/recipes/gemma4_31b/prune_local_heal.py
elif [[ -f /shared/dev/omnimergekit/recipes/gemma4_31b/prune_local_heal.py ]]; then
    RECIPE=/shared/dev/omnimergekit/recipes/gemma4_31b/prune_local_heal.py
else
    echo "ERROR: prune_local_heal.py not found. Pass RECIPE=<path> or clone omnimergekit." >&2
    exit 1
fi

mkdir -p "$WORKDIR"
CACHE_DIR="$WORKDIR/cache"
SRC_DIR="$WORKDIR/base"
OUT_DIR="$WORKDIR/pruned"
LOG="$WORKDIR/replay.log"

mkdir -p "$CACHE_DIR"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Iseconds)] === replay_prune ==="
echo "  base:         $BASE_HF_ID"
echo "  target:       $TARGET_HF_ID"
echo "  prune_frac:   $PRUNE_FRAC"
echo "  cache_src:    $CACHE_SRC"
echo "  workdir:      $WORKDIR"
echo "  recipe:       $RECIPE"
echo "  extra args:   ${EXTRA_ARGS[*]:-(none)}"

# --- Step 1: fetch cache artifacts ---
echo "[$(date -Iseconds)] step 1: fetching cache artifacts from $CACHE_SRC ..."
WANT_FILES=(
    gemma4_31b_imp_full_nf4.pt
    gemma4_31b_canary_baseline_n50.pt
    prune_manifest.json
    calibration_datav5.txt
)
if [[ "$CACHE_SRC" == *:* ]]; then
    # rsync syntax (host:path)
    for f in "${WANT_FILES[@]}"; do
        rsync -av --partial -e "ssh -o StrictHostKeyChecking=no" \
            "${CACHE_SRC%/}/$f" "$CACHE_DIR/" 2>&1 \
            || echo "  (optional: $f not found at source — continuing)"
    done
else
    # local path
    for f in "${WANT_FILES[@]}"; do
        if [[ -f "$CACHE_SRC/$f" ]]; then
            cp -v "$CACHE_SRC/$f" "$CACHE_DIR/" || true
        else
            echo "  (optional: $f not found at $CACHE_SRC — continuing)"
        fi
    done
fi
ls -la "$CACHE_DIR"

# Required: at minimum the importance cache must be present.
if [[ ! -s "$CACHE_DIR/gemma4_31b_imp_full_nf4.pt" ]]; then
    echo "ERROR: importance cache missing. Cannot replay without it." >&2
    echo "       Expected at $CACHE_SRC/gemma4_31b_imp_full_nf4.pt" >&2
    exit 1
fi
# Required: calibration corpus (prune_local_heal.py refuses to start without it
# on 31B — corpus is too large to embed and changes determinism).
CALIB_FILE="${CALIB_FILE:-$CACHE_DIR/calibration_datav5.txt}"
if [[ ! -s "$CALIB_FILE" ]]; then
    # Try the recipe's local sibling on this host as a last-resort fallback.
    for FALLBACK in \
        /workspace/omnimergekit/scripts/calibration_datav5.txt \
        /shared/dev/omnimergekit/scripts/calibration_datav5.txt \
        "$(dirname "$RECIPE")/../../scripts/calibration_datav5.txt"; do
        if [[ -s "$FALLBACK" ]]; then
            cp -v "$FALLBACK" "$CACHE_DIR/calibration_datav5.txt"
            CALIB_FILE="$CACHE_DIR/calibration_datav5.txt"
            break
        fi
    done
fi
if [[ ! -s "$CALIB_FILE" ]]; then
    echo "ERROR: calibration corpus not found. prune_local_heal.py --calib-file is mandatory." >&2
    echo "       Pass calibration_datav5.txt in the cache source dir, or set CALIB_FILE=<path>." >&2
    exit 1
fi
echo "  calib file: $CALIB_FILE ($(wc -c < "$CALIB_FILE") bytes)"
# Optional: canary baseline. Recipe will recapture it (~17 min) if missing.
HAVE_CANARY=""
if [[ -s "$CACHE_DIR/gemma4_31b_canary_baseline_n50.pt" ]]; then
    HAVE_CANARY=1
    echo "  canary baseline cache present — will reuse (saves ~17 min)"
else
    echo "  canary baseline cache missing — Phase 0 capture will run (~17 min)"
fi

# --- Step 2: download base model from HF ---
echo "[$(date -Iseconds)] step 2: downloading base $BASE_HF_ID to $SRC_DIR ..."
rm -rf "$SRC_DIR"
mkdir -p "$SRC_DIR"
HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$BASE_HF_ID" \
    --local-dir "$SRC_DIR" \
    --exclude '*.gguf' --exclude 'imatrix.dat'
echo "  base size: $(du -sh "$SRC_DIR" | cut -f1)"

# --- Step 3: run the prune via prune_local_heal.py with cache reuse ---
echo "[$(date -Iseconds)] step 3: running prune with cached artifacts ..."
rm -rf "$OUT_DIR"

PRUNE_ARGS=(
    --model-path "$SRC_DIR"
    --output     "$OUT_DIR"
    --calib-file "$CALIB_FILE"
    --prune-frac "$PRUNE_FRAC"
    --placement  auto
    --gpu-mem    "$GPU_MEM"
    --cpu-mem    "$CPU_MEM"
    --phase1-mode nf4_global
    --phase1-nf4-chunk-tokens 192
    --imp-cache  "$CACHE_DIR/gemma4_31b_imp_full_nf4.pt"
    --chunk-tokens "$CHUNK_TOKENS"
    --ridge      "$RIDGE"
)
if [[ -n "$HAVE_CANARY" ]]; then
    PRUNE_ARGS+=(--canary-baseline-cache "$CACHE_DIR/gemma4_31b_canary_baseline_n50.pt")
fi
PRUNE_ARGS+=("${EXTRA_ARGS[@]}")

echo "  invoking: python $RECIPE ${PRUNE_ARGS[*]}"
PYTORCH_ALLOC_CONF=expandable_segments:True python "$RECIPE" "${PRUNE_ARGS[@]}"

# --- Step 3.5: verify the produced model is not the .broken variant ---
if [[ ! -d "$OUT_DIR" ]]; then
    if [[ -d "${OUT_DIR}.broken" ]]; then
        echo "[$(date -Iseconds)] CANARY FAILED — output landed at ${OUT_DIR}.broken; NOT uploading."
        exit 2
    fi
    echo "ERROR: $OUT_DIR not present after prune." >&2
    exit 1
fi

echo "  pruned model size: $(du -sh "$OUT_DIR" | cut -f1)"
ls -la "$OUT_DIR"

# Cross-check head selection against the source manifest if present
if [[ -s "$CACHE_DIR/prune_manifest.json" && -s "$OUT_DIR/prune_manifest.json" ]]; then
    python3 - <<PY
import json, sys
src = json.load(open("$CACHE_DIR/prune_manifest.json"))
dst = json.load(open("$OUT_DIR/prune_manifest.json"))
def heads(m):
    # tolerant: pull whatever per-layer head-drop list the manifest uses
    for k in ("dropped_heads_per_layer", "drop_per_layer", "phase1c", "selection"):
        if k in m: return m[k]
    return None
sh, dh = heads(src), heads(dst)
print(f"  source heads: {'present' if sh else 'unparseable'}")
print(f"  dest heads:   {'present' if dh else 'unparseable'}")
if sh and dh and sh == dh:
    print("  ✓ head selection bit-identical to source manifest")
elif sh and dh:
    print("  ⚠ head selection differs from source manifest — investigate")
    sys.exit(0)
PY
fi

# --- Step 4: upload to HF unless suppressed ---
if [[ "${SKIP_UPLOAD:-0}" == "1" ]]; then
    echo "[$(date -Iseconds)] SKIP_UPLOAD=1 — leaving $OUT_DIR on disk."
    echo "  Local path: $OUT_DIR"
    exit 0
fi

echo "[$(date -Iseconds)] step 4: uploading $OUT_DIR → $TARGET_HF_ID ..."
hf repo create "$TARGET_HF_ID" --type model -y 2>&1 | tail -3 || true
HF_HUB_ENABLE_HF_TRANSFER=1 hf upload "$TARGET_HF_ID" "$OUT_DIR" . \
    --commit-message "replay prune from cached importance (frac=$PRUNE_FRAC) of $BASE_HF_ID"

# Free disk
rm -rf "$SRC_DIR"
echo "[$(date -Iseconds)] === DONE ==="
echo "  https://huggingface.co/$TARGET_HF_ID"
