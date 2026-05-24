#!/usr/bin/env bash
# bootstrap_reeval.sh — thin wrapper over eval/pod_eval_bootstrap.sh.
#
# Variant: NVFP4A16 re-eval / re-quant pod (Gemma 4). Full requirements-based
# omnimergekit env + the modelopt env (NVFP4A16 builder) + Fix-A lm-eval patch
# for the vLLM reasoning-content silent-empty pathology. Pulls base 128e + a
# pruned variant. No llama.cpp (vLLM path).
#
# NOTE vs the old one-off: modelopt is now pinned to 0.43.0 (the canonical pin,
# feedback_modelopt_pin_0_43) instead of the old HEAD 87ea8babe — intentional.
#
# Run on the pod:  HF_TOKEN=... bash bootstrap_reeval.sh [extra flags...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../pod_eval_bootstrap.sh" \
    --no-llama \
    --deps eval-full --patches unbound,fix-a \
    --modelopt-env \
    --hf-pull "google/gemma-4-26B-A4B-it:/workspace/base-128e ManniX-ITA/gemma-4-A4B-98e-v4-it:/workspace/v4" \
    "$@"
