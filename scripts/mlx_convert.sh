#!/usr/bin/env bash
# Convert a Hugging Face model to MLX-quantized format and upload it back to
# Hugging Face. Auto-detects multimodal vs text-only from the source
# `config.json` and routes through `mlx_vlm.convert` or `mlx_lm.convert`
# accordingly. Designed to work on Linux + NVIDIA via the `mlx-cuda` backend,
# which is the only way to run MLX outside Apple Silicon today.
#
# This is the canonical OmniMergeKit MLX-conversion runner. See MLX_CONVERT.md
# in this directory for usage + the version-pinning rationale.
#
# Usage:
#   bash mlx_convert.sh <hf_source_repo> <hf_target_repo> [bits] [extra mlx_(vlm|lm).convert args]
#
# Auto-routing rule:
#   - if source `config.json` has `vision_config` set, OR `architectures`
#     contains "VL" / "Vision" / "ForConditionalGeneration"
#       → mlx_vlm.convert (preserves vision tower + processors)
#   - else
#       → mlx_lm.convert (text-only, smaller output)
#
# Override the auto-detection with $MLX_BACKEND=vlm or $MLX_BACKEND=lm
# (handy for VLM bases you want to ship as text-only or vice versa).
#
# Examples:
#   # text-only (auto)
#   bash mlx_convert.sh ManniX-ITA/Qwen3.6-27B-Omnimerge-v4 \
#                       ManniX-ITA/Qwen3.6-27B-Omnimerge-v4-MLX-4bit 4
#   # multimodal (auto: source has vision_config)
#   bash mlx_convert.sh Qwen/Qwen3-VL-30B-A3B-Instruct \
#                       ManniX-ITA/Qwen3-VL-30B-A3B-MLX-4bit 4
#   # force VLM path on an autodetect-text source (e.g. a merge that
#   # stripped vision_config but still has the projector weights):
#   MLX_BACKEND=vlm bash mlx_convert.sh src target 4
#
# Required env:
#   HF_TOKEN — write token for the target repo
#
# Optional env:
#   MLX_BACKEND — force "lm" or "vlm" (default: auto-detect)
#   WORKDIR     — staging dir (default /workspace/mlx_convert)
#   ENV_NAME    — conda env name (default `mlx`)
#   CONDA_ROOT  — conda root (default /opt/conda)
#
# Output: weights + tokenizer + (processor + vision configs when multimodal)
# land at /workspace/mlx_convert/<target>/mlx and get `hf upload`-ed.
set -euo pipefail
SOURCE="${1:?usage: $0 <hf_source_repo> <hf_target_repo> [bits] [extra args...]}"
TARGET="${2:?usage: $0 <hf_source_repo> <hf_target_repo> [bits] [extra args...]}"
BITS="${3:-4}"
shift 3 2>/dev/null || shift $#
EXTRA_ARGS=("$@")

: "${HF_TOKEN:?HF_TOKEN env var required (write token for the target repo)}"
: "${WORKDIR:=/workspace/mlx_convert}"
: "${ENV_NAME:=mlx}"
: "${CONDA_ROOT:=/opt/conda}"
: "${MLX_BACKEND:=auto}"   # auto | lm | vlm

# --- ensure conda env exists with the known-working pin ---
# Why these pins (verified 2026-05-11 on Linux + RTX 3090 + CUDA 12.1):
#   - mlx==0.30.0 and mlx-cuda==0.30.0 must match exactly (ABI-coupled).
#   - mlx-lm==0.30.7 is the lower bound for Qwen3.5/3.6 conversion
#     (qwen3_5.py first ships in 0.30; prior versions reject model_type
#     "qwen3_5"). The previous doc warning that mlx-lm 0.27+ needs
#     mlx>=0.31 only applies to inference-path symbols, NOT convert.
#   - mlx-vlm (latest) for multimodal conversion. Imports cleanly against
#     mlx-cuda 0.30; preserves vision_tower, multi_modal_projector,
#     preprocessor_config, video_preprocessor_config, chat_template.
#   - mlx-cuda installs nvidia-cublas-cu12 12.9 + nvidia-cuda-nvrtc-cu12 12.9
#     which conflicts with pytorch 2.5.1+cu121 — keep MLX in its own env.
#   - Inference on the CUDA build currently fails (cooperative_groups SDK
#     mismatch) but CONVERSION works fine and that's all we need; end users
#     run the output on Apple Silicon natively.
if [[ ! -d "$CONDA_ROOT/envs/$ENV_NAME" ]]; then
    echo "[$(date -Iseconds)] creating conda env '$ENV_NAME' ..."
    "$CONDA_ROOT/bin/conda" create -y -n "$ENV_NAME" python=3.11
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# mlx-lm is always required (vlm depends on it); install mlx-vlm lazily only
# when we need the VLM path so single-purpose text-only pods don't pull it.
if ! python -c "import mlx_lm" >/dev/null 2>&1; then
    echo "[$(date -Iseconds)] installing pinned mlx-lm stack ..."
    pip install -q --upgrade pip
    pip install -q "mlx==0.30.0" "mlx-cuda==0.30.0" "mlx-lm==0.30.7"
fi

# --- preflight: sanity-check mlx import + GPU stream ---
python - <<'EOF'
import mlx.core as mx
with mx.stream(mx.gpu):
    x = mx.array([1.0]) + mx.array([2.0])
mx.eval(x)
assert x.tolist() == [3.0], "mlx GPU smoke failed"
print("[mlx_convert] mlx import + GPU stream OK")
EOF

# --- workspace layout ---
TARGET_SLUG=$(echo "$TARGET" | tr '/' '_')
SRC_DIR="$WORKDIR/$TARGET_SLUG/source"
MLX_DIR="$WORKDIR/$TARGET_SLUG/mlx"
mkdir -p "$SRC_DIR"

LOG="$WORKDIR/$TARGET_SLUG/convert.log"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Iseconds)] === mlx_convert.sh ==="
echo "  source:      $SOURCE"
echo "  target:      $TARGET"
echo "  bits:        $BITS"
echo "  backend hint: $MLX_BACKEND"
echo "  extra:       ${EXTRA_ARGS[*]:-(none)}"

# --- download source weights (no GGUF, no imatrix) ---
echo "[$(date -Iseconds)] downloading source weights ..."
HF_HUB_ENABLE_HF_TRANSFER=1 \
    hf download "$SOURCE" \
        --local-dir "$SRC_DIR" \
        --exclude '*.gguf' --exclude 'imatrix.dat'

# --- detect backend ---
if [[ "$MLX_BACKEND" == "auto" ]]; then
    BACKEND=$(python - <<EOF
import json
cfg = json.load(open("$SRC_DIR/config.json"))
archs = str(cfg.get("architectures") or [])
is_vlm = bool(cfg.get("vision_config")) or any(
    s in archs for s in ("VL", "Vision", "ForConditionalGeneration")
)
print("vlm" if is_vlm else "lm")
EOF
    )
else
    BACKEND="$MLX_BACKEND"
fi
echo "[$(date -Iseconds)] backend selected: $BACKEND"

# Lazily install mlx-vlm if needed
if [[ "$BACKEND" == "vlm" ]] && ! python -c "import mlx_vlm" >/dev/null 2>&1; then
    echo "[$(date -Iseconds)] installing mlx-vlm (multimodal path) ..."
    pip install -q "mlx-vlm"
fi

# --- convert ---
echo "[$(date -Iseconds)] converting to MLX with $BITS-bit quantization (backend=$BACKEND) ..."
CONVERT_ARGS=(--hf-path "$SRC_DIR" --mlx-path "$MLX_DIR")
if [[ "$BITS" =~ ^(4|8)$ ]]; then
    CONVERT_ARGS+=(-q --q-bits "$BITS")
    # mlx-vlm uses --q-group-size; mlx-lm defaults to 64 with no flag
    if [[ "$BACKEND" == "vlm" ]]; then
        CONVERT_ARGS+=(--q-group-size 64)
    fi
fi
# Ensure target dir doesn't exist (both converters refuse to overwrite)
rm -rf "$MLX_DIR"
if [[ "$BACKEND" == "vlm" ]]; then
    python -m mlx_vlm.convert "${CONVERT_ARGS[@]}" "${EXTRA_ARGS[@]}"
else
    mlx_lm.convert "${CONVERT_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi

# --- verify ---
echo "[$(date -Iseconds)] verifying output ..."
python - <<EOF
import json, os, glob
cfg_p = "$MLX_DIR/config.json"
cfg = json.load(open(cfg_p))
q = cfg.get("quantization")
print(f"  config.quantization: {q}")
print(f"  model_type: {cfg.get('model_type')}")
print(f"  hidden_size: {cfg.get('hidden_size')}")
print(f"  vision_config: {'present' if cfg.get('vision_config') else 'absent'}")
sf_idx = os.path.join("$MLX_DIR", "model.safetensors.index.json")
sf_single = os.path.join("$MLX_DIR", "model.safetensors")
sf_glob = sorted(glob.glob(os.path.join("$MLX_DIR", "model-*.safetensors")))
if sf_glob:
    total = sum(os.path.getsize(p) for p in sf_glob) / 1e9
    print(f"  safetensors: {len(sf_glob)} shards, total {total:.2f} GB")
elif os.path.exists(sf_single):
    print(f"  safetensors: {os.path.getsize(sf_single)/1e9:.2f} GB")
else:
    print("  WARN: no safetensors output")
# Multimodal sidecars (relevant when backend=vlm)
for kind in ("preprocessor_config.json", "video_preprocessor_config.json",
             "chat_template.jinja", "processor_config.json"):
    p = os.path.join("$MLX_DIR", kind)
    if os.path.exists(p):
        print(f"  sidecar present: {kind}")
EOF

# --- upload ---
echo "[$(date -Iseconds)] creating HF repo (if missing) + uploading ..."
hf repo create "$TARGET" --type model -y 2>&1 | tail -3 || true
HF_HUB_ENABLE_HF_TRANSFER=1 hf upload "$TARGET" "$MLX_DIR" . \
    --commit-message "MLX-${BITS}bit (${BACKEND}) conversion of $SOURCE"

echo "[$(date -Iseconds)] === DONE ==="
echo "  url: https://huggingface.co/$TARGET"
