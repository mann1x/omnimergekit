#!/usr/bin/env bash
# bootstrap_hep_sweep.sh — thin wrapper over eval/pod_eval_bootstrap.sh.
#
# Variant: HE+ llama.cpp sweep on a fresh cuda:12.8 pod. Builds llama.cpp with
# LLAMA_CURL=ON (this variant pulls GGUFs over the network), full
# requirements-based omnimergekit env, plus the extra apt deps the sweep needs.
#
# Run on the pod:  bash bootstrap_hep_sweep.sh [extra flags...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../pod_eval_bootstrap.sh" \
    --llama --llama-curl --cuda-arch 86 \
    --apt-extra "libcurl4-openssl-dev pkg-config zstd lshw" \
    --deps eval-full \
    "$@"
