#!/usr/bin/env bash
# setup_conda_envs.sh — on-pod conda env recipes, single-sourced.
#
# This is the pod-SIDE body of the canonical eval-env setup. It is sourced by
# BOTH:
#   - scripts/pod_setup_eval_envs.sh  (controller wrapper: ssh in, source this)
#   - eval/pod_eval_bootstrap.sh      (on-pod: source this for --vllm-env /
#                                      --modelopt-env / --deps eval-full)
#
# Keeping the version-pinned recipes in ONE place stops the "five divergent
# bootstraps" drift (T111). Pins are authoritative — do NOT bump without an
# explicit cohort re-eval decision:
#   - modelopt 0.43.0  (feedback_modelopt_pin_0_43 — 0.44 NaNs on Gemma 4 bf16)
#   - transformers 5.8.0 in modelopt env (modelopt needs newer than omk's 5.5.0)
#   - torch 2.10.0+cu128, vllm 0.20.2  (match solidPC stack@2)
#
# Source it, then call the functions you need:
#   source setup_conda_envs.sh
#   ensure_omk_env [ENV_NAME]          # default omnimergekit; requirements-based
#   ensure_modelopt_env
#   ensure_vllm_env [VERSION]          # default 0.20.2
#   apply_lmeval_patches ENV_PY "unbound,fix-a"
#
# Assumes: /workspace/miniconda present + sourced, /workspace/omnimergekit cloned.
# Idempotent throughout.

# Resolve repo paths relative to THIS file (eval/pod_runners/setup_conda_envs.sh).
_SCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="$(cd "$_SCE_DIR/../patches" && pwd)"
REPO_ROOT="$(cd "$_SCE_DIR/../.." && pwd)"
CONDA_ENVS="${CONDA_ENVS:-/workspace/miniconda/envs}"

_sce_log() { echo "[$(date -Iseconds)] [envs] $*"; }

# Accept the two anaconda.com channel ToS (conda 26.x refuses create otherwise).
_sce_accept_tos() {
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>&1 | tail -1 || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>&1 | tail -1 || true
}

# apply_lmeval_patches <env_python> <patches_csv>
# patches_csv ∈ {unbound, fix-a} (comma-separated). No-op for unknown tokens.
apply_lmeval_patches() {
    local env_py="$1" csv="${2:-}"
    [ -n "$csv" ] || return 0
    local lm_dir
    lm_dir="$("$env_py" -c 'import lm_eval,os;print(os.path.dirname(lm_eval.__file__))' 2>/dev/null)" || {
        _sce_log "apply_lmeval_patches: lm_eval not importable in $env_py — skip"; return 0; }
    local IFS=,
    for p in $csv; do
        case "$p" in
            unbound)
                "$env_py" "$PATCHES_DIR/lm_eval_unbound_guard.py" "$lm_dir/models/api_models.py" ;;
            fix-a)
                "$env_py" "$PATCHES_DIR/fix_a_lm_eval_patch.py" "$lm_dir/models/openai_completions.py" ;;
            *) _sce_log "unknown patch '$p' — skip" ;;
        esac
    done
    find "$lm_dir" -name "*.pyc" -delete 2>/dev/null || true
    find "$lm_dir" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
}

# ensure_omk_env [ENV_NAME] — full requirements-based eval env (lm-eval,
# transformers 5.5.0). Default name omnimergekit. Applies the unbound patch.
ensure_omk_env() {
    local name="${1:-omnimergekit}" pfx
    pfx="$CONDA_ENVS/$name"
    if [ ! -f "$pfx/bin/lm-eval" ]; then
        _sce_log "create $name env (requirements-based)"
        _sce_accept_tos
        conda create -n "$name" python=3.11 -y 2>&1 | tail -3
        "$pfx/bin/pip" install --quiet --upgrade pip wheel setuptools
        # causal-conv1d / flash-linear-attention are Qwen3.5/3.6 *merge-engine*
        # deps (SSM CUDA kernels) that build from source and BREAK pip metadata
        # generation on a bare CUDA image (2026-05-27 causal-conv1d FATAL on
        # pod 38081385). Never needed for llama.cpp/vLLM eval — keep filtered.
        #
        # bitsandbytes USED TO be filtered alongside them as if it were an
        # SSM kernel — that was a category error. It ships as clean pip wheels
        # (no from-source build), and Router-KD's PagedAdamW8bit + the NF4
        # 4-bit student loading both REQUIRE it. Removing it from the filter
        # so eval-stack pods can run Router-KD without manual pip-install.
        # Origin: T141 on linode-blackswan-2 (2026-05-28) — Router-KD chain
        # exited 0 GPU-seconds in with `FATAL: python module bitsandbytes
        # missing`.
        local req_eval="$REPO_ROOT/.requirements_eval_filtered.txt"
        grep -vE '^(causal-conv1d|flash-linear-attention)' \
            "$REPO_ROOT/requirements.txt" > "$req_eval"
        "$pfx/bin/pip" install --quiet --no-build-isolation -r "$req_eval"   2>&1 | tail -5
        "$pfx/bin/pip" install --quiet -r "$REPO_ROOT/requirements-eval.txt" 2>&1 | tail -3
    else
        _sce_log "$name env exists"
    fi
    apply_lmeval_patches "$pfx/bin/python" "unbound"
    "$pfx/bin/python" -c "import lm_eval, transformers; print(f'  omk env: lm_eval={lm_eval.__version__} transformers={transformers.__version__}')"
}

# ensure_modelopt_env — NVFP4A16 quant builder env (quantize_any.py).
ensure_modelopt_env() {
    local pfx="$CONDA_ENVS/modelopt"
    if [ ! -f "$pfx/bin/python" ]; then
        _sce_log "create modelopt env (modelopt 0.43.0 / transformers 5.8.0 / torch cu128)"
        _sce_accept_tos
        conda create -n modelopt python=3.11 -y 2>&1 | tail -3
        "$pfx/bin/pip" install --quiet --upgrade pip wheel setuptools
        "$pfx/bin/pip" install --quiet --index-url https://download.pytorch.org/whl/cu128 'torch==2.10.0+cu128' 2>&1 | tail -2
        "$pfx/bin/pip" install --quiet \
            'nvidia-modelopt==0.43.0' 'transformers==5.8.0' 'accelerate>=1.13.0' \
            'hf-transfer>=0.1.6' 'huggingface_hub>=0.24' 'datasets>=3.0' 'safetensors>=0.4' 2>&1 | tail -3
        mkdir -p /root/.conda
        grep -qxF "$pfx" /root/.conda/environments.txt 2>/dev/null || echo "$pfx" >> /root/.conda/environments.txt
    else
        _sce_log "modelopt env exists"
    fi
    "$pfx/bin/python" -c "import modelopt, transformers, torch; print(f'  modelopt env: modelopt={modelopt.__version__} transformers={transformers.__version__} torch={torch.__version__}')"
}

# ensure_vllm_env [VERSION] — vLLM server env. Default 0.20.2.
ensure_vllm_env() {
    local ver="${1:-0.20.2}" pfx="$CONDA_ENVS/vllm"
    if [ ! -d "$pfx" ]; then
        _sce_log "create vllm env (vllm==$ver)"
        _sce_accept_tos
        conda create -n vllm python=3.11 -y 2>&1 | tail -3
        "$pfx/bin/pip" install --quiet --upgrade pip wheel setuptools
        "$pfx/bin/pip" install --quiet "vllm==$ver" 2>&1 | tail -5
    else
        _sce_log "vllm env exists"
    fi
    "$pfx/bin/python" -c "import vllm; print(f'  vllm env: {vllm.__version__}')"
}
