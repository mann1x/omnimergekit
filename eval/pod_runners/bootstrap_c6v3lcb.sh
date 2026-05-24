#!/usr/bin/env bash
# bootstrap_c6v3lcb.sh — thin wrapper over eval/pod_eval_bootstrap.sh.
#
# Variant: Gemma 4 26B-A4B v6-coder C6v3lcb, llama.cpp Q6_K canonical 9-bench
# eval on a 2x3090 vast pod whose image ships a torch env named `rr`.
# Builds llama.cpp (arch 86), augments `rr` with the lean lm-eval stack, and
# lays the solidpc-path symlink farm so the hardcoded eval_suite_llama.sh runs.
#
# Run on the pod:  HF_TOKEN=... bash bootstrap_c6v3lcb.sh [extra flags...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../pod_eval_bootstrap.sh" \
    --llama --cuda-arch 86 \
    --env rr --deps eval-augment --patches unbound \
    --symlink-farm \
    --tokenizer-link /workspace/base-128e:/workspace/google/gemma-4-26B-A4B-it \
    "$@"
