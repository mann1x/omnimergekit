#!/usr/bin/env bash
# bootstrap_router_recovery.sh — thin wrapper over eval/pod_eval_bootstrap.sh.
#
# Variant: T18 router-recovery (EAC + Router-KD) tooling validation on a 2x3090
# vast pod. No eval/llama.cpp — this is a TRAINING/surgery env (torch + bnb +
# datasets + accelerate). Installs miniconda on a bare image, creates env `rr`,
# pulls the base 128e + a pruned variant (bf16 safetensors only; skips bundled
# gguf). Edit --hf-pull to retarget the pruned variant.
#
# Run on the pod:  HF_TOKEN=... bash bootstrap_router_recovery.sh [extra flags...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../pod_eval_bootstrap.sh" \
    --no-llama --install-miniconda \
    --env rr --deps train \
    --hf-pull "google/gemma-4-26B-A4B-it:/workspace/base-128e ManniX-ITA/gemma-4-A4B-98e-v5-coder-it:/workspace/v5coder" \
    --hf-pull-patterns "*.safetensors *.json tokenizer* *.model *.jinja" \
    "$@"
