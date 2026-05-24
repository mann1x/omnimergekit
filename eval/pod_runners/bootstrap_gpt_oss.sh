#!/usr/bin/env bash
# bootstrap_gpt_oss.sh — thin wrapper over eval/pod_eval_bootstrap.sh.
#
# Variant: openai/gpt-oss-20b (native MXFP4, ~14 GB) on a 24 GB pod, vLLM path.
# Full requirements-based omnimergekit env + the vllm env (0.20.2, Harmony /
# gpt-oss support). No llama.cpp, no modelopt (model is already quantized).
#
# Run on the pod:  HF_TOKEN=... bash bootstrap_gpt_oss.sh [extra flags...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../pod_eval_bootstrap.sh" \
    --no-llama \
    --deps eval-full \
    --vllm-env \
    --hf-pull "openai/gpt-oss-20b:/workspace/google/openai__gpt-oss-20b" \
    --hf-pull-patterns "*.json *.safetensors tokenizer* *.txt *.md *.py" \
    "$@"
