#!/bin/bash
set -e
cd /srv/ml/repos
rm -rf llama.cpp-latest
git clone --depth 1 https://github.com/ggml-org/llama.cpp llama.cpp-latest
cd llama.cpp-latest
echo "HEAD: $(git log -1 --format=%H\ %ci)"
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_FA=ON -DLLAMA_CURL=OFF
cmake --build build -j 32 --target llama-server llama-cli llama-bench
echo "=== BUILD DONE $(date -Iseconds) ==="
ls -la build/bin/llama-server build/bin/llama-cli build/bin/llama-bench
