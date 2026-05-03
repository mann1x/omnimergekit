#!/bin/bash
# One-shot setup for a fresh vast.ai pod to run the Qwen3.5-27B Omnimerge pipeline.
# Installs: build deps, cmake, llama.cpp (with CUDA), mergekit, python deps, HF login.
#
# Usage (on pod, as root):
#   HF_TOKEN=hf_xxx bash pod_qwen3_5_setup.sh
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN env var required}"

echo "===== $(date) POD SETUP START ====="

# --- System deps ---
apt-get update -qq
apt-get install -y -qq cmake build-essential git curl

# --- Python deps for HF + llama.cpp + mergekit ---
# NOTE: upgrading torchvision is critical — conda image ships torch/torchvision mismatch
# that breaks lm-eval humaneval task with 'torchvision::nms does not exist' error.
# Forcing `pip install --upgrade torch torchvision` pulls a matched pair.
pip install -q --upgrade \
    huggingface_hub \
    "transformers>=4.51" \
    sentencepiece protobuf \
    numpy safetensors accelerate \
    "mergekit>=0.0.5" \
    torch torchvision

# --- HF login ---
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False); print('HF login OK')"

# --- llama.cpp build with CUDA ---
if [[ ! -x /opt/llama.cpp/build/bin/llama-quantize ]]; then
    if [[ ! -d /opt/llama.cpp ]]; then
        git clone --depth=1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp
    fi
    cd /opt/llama.cpp
    pip install -q -r requirements.txt || true
    export PATH=/usr/local/cuda/bin:$PATH
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    cmake --build build --config Release -j$(nproc)
    echo "llama.cpp built."
else
    echo "llama.cpp already built."
fi

# --- lm-evaluation-harness ---
if ! command -v lm_eval >/dev/null 2>&1; then
    pip install -q "lm-eval[api]"
fi

# --- libstdc++ fix for conda-based pytorch images ---
# vast.ai pytorch/pytorch:*-cuda*-cudnn*-devel ships conda libstdc++ 3.4.29 which is older
# than the system libstdc++ against which llama.cpp gets built. Replace conda's symlink.
if [[ -L /opt/conda/lib/libstdc++.so.6 ]] && \
   ! /opt/llama.cpp/build/bin/llama-quantize --help 2>&1 | grep -q "Quantize"; then
    echo "  fixing conda libstdc++ symlink → system lib..."
    rm /opt/conda/lib/libstdc++.so.6
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6
fi

echo "===== $(date) POD SETUP DONE ====="
echo "tools:"
which python3 lm_eval
ls /opt/llama.cpp/build/bin/ | head
