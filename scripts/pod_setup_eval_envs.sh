#!/bin/bash
# pod_setup_eval_envs.sh — replicate solidPC's `omnimergekit` + `vllm` conda
# envs on the pod, per `docs/CONDA_ENVS.md`.
#
# After this runs, /root/anaconda3/envs/{omnimergekit,vllm} will exist with
# the documented requirements (single-env design: omnimergekit env has
# requirements.txt + requirements-eval.txt; the separate `vllm` env carries
# the patched vllm-source build).
set -euo pipefail

POD_HOST="${POD_HOST:-ssh9.vast.ai}"
POD_PORT="${POD_PORT:-35692}"
POD_USER="${POD_USER:-root}"
SSH="ssh -o StrictHostKeyChecking=no -p $POD_PORT $POD_USER@$POD_HOST"

$SSH bash <<'POD'
set -euo pipefail
source /workspace/miniconda/etc/profile.d/conda.sh
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_ENABLE_HF_TRANSFER=1

# eval_suite_vllm.sh hardcodes /root/anaconda3/envs/* — symlink so it finds
# our envs.
mkdir -p /root/anaconda3
[ -e /root/anaconda3/envs ] || ln -s /workspace/miniconda/envs /root/anaconda3/envs

# ── omnimergekit env (lm-eval, transformers 5.5.0) ─────────────────────────
if [ ! -f /workspace/miniconda/envs/omnimergekit/bin/lm-eval ]; then
    echo "[$(date +%H:%M:%S)] create omnimergekit env"
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>&1 | tail -1 || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>&1 | tail -1 || true
    conda create -n omnimergekit python=3.11 -y 2>&1 | tail -3

    /workspace/miniconda/envs/omnimergekit/bin/pip install --quiet --upgrade pip wheel setuptools
    # Install core + eval requirements per docs/CONDA_ENVS.md
    /workspace/miniconda/envs/omnimergekit/bin/pip install --quiet --no-build-isolation -r /workspace/omnimergekit/requirements.txt 2>&1 | tail -5
    /workspace/miniconda/envs/omnimergekit/bin/pip install --quiet -r /workspace/omnimergekit/requirements-eval.txt 2>&1 | tail -3
else
    echo "[$(date +%H:%M:%S)] omnimergekit env exists"
fi
/workspace/miniconda/envs/omnimergekit/bin/python -c "import lm_eval, transformers; print(f'omk env: lm_eval={lm_eval.__version__} transformers={transformers.__version__}')"

# ── lm-eval api_models.py patch: UnboundLocalError on retry exhaust ───────
# Upstream lm-eval 0.4.11 has `outputs` referenced in `except BaseException`
# without being initialized — if `session.post(...)` raises before
# `outputs = await response.json()` lands (e.g. tenacity retries exhaust on
# transient timeouts), the except handler crashes with `UnboundLocalError:
# cannot access local variable 'outputs'`. The error abort propagates as
# task-level FAIL even though all completed samples are already in the
# sqlite cache. Hit on pod 36755693 (2026-05-14) on gsm8k/gpqa/etc tail-end
# completions. See memory/feedback_lm_eval_unbound_outputs_bug.md.
LMEVAL_API=/workspace/miniconda/envs/omnimergekit/lib/python3.11/site-packages/lm_eval/models/api_models.py
if [ -f "$LMEVAL_API" ] && ! grep -q "outputs = None  # PATCH" "$LMEVAL_API"; then
    /workspace/miniconda/envs/omnimergekit/bin/python - <<'PY'
import pathlib
p = pathlib.Path("/workspace/miniconda/envs/omnimergekit/lib/python3.11/site-packages/lm_eval/models/api_models.py")
src = p.read_text()
marker = '        cache_method = "generate_until" if generate else "loglikelihood"\n        acquired = await sem.acquire()\n        try:'
patched = '        cache_method = "generate_until" if generate else "loglikelihood"\n        outputs = None  # PATCH: avoid UnboundLocalError in except\n        acquired = await sem.acquire()\n        try:'
if marker in src:
    p.write_text(src.replace(marker, patched, 1))
    print("[lm-eval patch] api_models.py: outputs=None guard inserted")
else:
    print("[lm-eval patch] MARKER NOT FOUND — lm-eval version drift?")
PY
    # Purge stale .pyc and prevent fresh ones (PYTHONDONTWRITEBYTECODE)
    find /workspace/miniconda/envs/omnimergekit/lib/python3.11/site-packages/lm_eval -name "*.pyc" -delete 2>/dev/null
    find /workspace/miniconda/envs/omnimergekit/lib/python3.11/site-packages/lm_eval -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
fi

# ── lm-eval Fix-A: reasoning_content fallback for openai_completions ──────
# When vLLM 0.20.2 wheel (unpatched — without #42250 closure-capture fix) serves
# Gemma 4 with reasoning_parser enabled, ~98% of chat completions return
# `content=""` and put the full output in `reasoning_content`. lm-eval's
# openai_completions.parse_generations only reads `content` → empty samples.
# On pod 36755693 (2026-05-14) this gave arc_challenge_reeval24k 57/58 empty
# (1.72% pass) and gsm8k_reeval24k 3/7 empty even with the cached results.
# Fix: fall back to reasoning_content when content is empty. Idempotent.
# See memory/feedback_vllm_gemma4_silentempty_rca.md.
LMEVAL_CHAT=/workspace/miniconda/envs/omnimergekit/lib/python3.11/site-packages/lm_eval/models/openai_completions.py
if [ -f "$LMEVAL_CHAT" ] && [ -f /workspace/scripts/fix_a_lm_eval_patch.py ]; then
    python3 /workspace/scripts/fix_a_lm_eval_patch.py "$LMEVAL_CHAT"
    find /workspace/miniconda/envs/omnimergekit/lib/python3.11/site-packages/lm_eval -name "*.pyc" -delete 2>/dev/null
fi

# ── modelopt env (NVFP4A16 quant builder via quantize_any.py) ─────────────
# Required by pod_quant_31b.sh / pod_quant_*.sh / any quantize_any.py run.
# Pinned versions per memory/feedback_modelopt_pin_0_43.md:
#   - nvidia-modelopt==0.43.0 (0.44+ NaN on Gemma 4 bf16 fused-MoE calibration;
#                              config dict→list regression)
#   - transformers==5.8.0     (modelopt requires newer than omk's 5.5.0)
#   - torch==2.10.0+cu128     (matches solidPC; cu128 because some pod hosts
#                              have CUDA 12.8 nvcc but old cuDNN that bricks
#                              older torch — see feedback/pod_image_requirements.md)
# Disk: ~3 GB env footprint. Setup time: 5-10 min on a fresh pod.
# Pod 36755693 (2026-05-14) hit this gap: pod_quant_31b.sh did `conda activate
# modelopt` which silently failed because the env was never created → quant
# phase finished in 1 sec with exit 0 (chain reported all green). See
# memory/feedback_pod_modelopt_env_missing.md.
if [ ! -f /workspace/miniconda/envs/modelopt/bin/python ]; then
    echo "[$(date +%H:%M:%S)] create modelopt env"
    conda create -n modelopt python=3.11 -y 2>&1 | tail -3
    /workspace/miniconda/envs/modelopt/bin/pip install --quiet --upgrade pip wheel setuptools
    /workspace/miniconda/envs/modelopt/bin/pip install --quiet \
        --index-url https://download.pytorch.org/whl/cu128 'torch==2.10.0+cu128' 2>&1 | tail -2
    /workspace/miniconda/envs/modelopt/bin/pip install --quiet \
        'nvidia-modelopt==0.43.0' \
        'transformers==5.8.0' \
        'accelerate>=1.13.0' \
        'hf-transfer>=0.1.6' \
        'huggingface_hub>=0.24' \
        'datasets>=3.0' \
        'safetensors>=0.4' 2>&1 | tail -3
    # Register name so `conda activate modelopt` works in plain bash
    mkdir -p /root/.conda
    grep -qxF "/workspace/miniconda/envs/modelopt" /root/.conda/environments.txt 2>/dev/null \
        || echo "/workspace/miniconda/envs/modelopt" >> /root/.conda/environments.txt
else
    echo "[$(date +%H:%M:%S)] modelopt env exists"
fi
/workspace/miniconda/envs/modelopt/bin/python -c "
import modelopt, transformers, torch
print(f'modelopt env: modelopt={modelopt.__version__} transformers={transformers.__version__} torch={torch.__version__}')
"

# ── vllm env (vLLM server with patched source) ─────────────────────────────
if [ ! -f /workspace/miniconda/envs/vllm/bin/vllm ] && [ ! -d /workspace/miniconda/envs/vllm ]; then
    echo "[$(date +%H:%M:%S)] create vllm env"
    conda create -n vllm python=3.11 -y 2>&1 | tail -3
    /workspace/miniconda/envs/vllm/bin/pip install --quiet --upgrade pip wheel setuptools

    # Use precompiled vllm wheel (matches solidPC's 0.20.2 version family)
    # then overlay our patched source by adding vllm-source to PYTHONPATH in
    # the launcher. The wheel install pulls all C-ext deps cleanly.
    /workspace/miniconda/envs/vllm/bin/pip install --quiet vllm==0.20.2 2>&1 | tail -5
fi
/workspace/miniconda/envs/vllm/bin/python -c "import vllm; print(f'vllm env: {vllm.__version__}')"

echo "[$(date +%H:%M:%S)] DONE — both envs ready"
POD
