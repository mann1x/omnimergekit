#!/bin/bash
# pod_setup_eval_envs.sh — controller-side wrapper: replicate solidPC's
# `omnimergekit` + `modelopt` + `vllm` conda envs on a pod by ssh'ing in and
# sourcing the canonical on-pod env recipes (eval/pod_runners/setup_conda_envs.sh).
#
# The version pins (modelopt 0.43.0, transformers 5.8.0, torch 2.10.0+cu128,
# vllm 0.20.2) live in that ONE lib, shared with eval/pod_eval_bootstrap.sh —
# this script just delivers them over ssh. See EVAL_PROTOCOL.md §2.5.
#
# Assumes the pod already has /workspace/miniconda and a /workspace/omnimergekit
# clone (git or rsync). For a from-scratch pod (apt, miniconda, llama.cpp, repo,
# symlink farm, HF pulls), use the on-pod eval/pod_eval_bootstrap.sh instead.
set -euo pipefail

POD_HOST="${POD_HOST:-ssh9.vast.ai}"
POD_PORT="${POD_PORT:-35692}"
POD_USER="${POD_USER:-root}"
SSH="ssh -o StrictHostKeyChecking=no -p $POD_PORT $POD_USER@$POD_HOST"

$SSH bash <<'POD'
set -euo pipefail
source /workspace/miniconda/etc/profile.d/conda.sh
export PYTHONDONTWRITEBYTECODE=1
export HF_XET_HIGH_PERFORMANCE=1

# eval_suite_*.sh hardcode /root/anaconda3/envs/* — symlink so they resolve.
mkdir -p /root/anaconda3
[ -e /root/anaconda3/envs ] || ln -s /workspace/miniconda/envs /root/anaconda3/envs

LIB=/workspace/omnimergekit/eval/pod_runners/setup_conda_envs.sh
if [ ! -f "$LIB" ]; then
    echo "FATAL: $LIB missing — clone/rsync omnimergekit to /workspace/omnimergekit first"; exit 1
fi
# shellcheck disable=SC1090
source "$LIB"

ensure_omk_env omnimergekit
# ensure_omk_env applies the 'unbound' guard; the controller path also applies
# Fix-A (vLLM Gemma 4 reasoning_content silent-empty fallback).
apply_lmeval_patches /workspace/miniconda/envs/omnimergekit/bin/python "fix-a"
ensure_modelopt_env
ensure_vllm_env

echo "[$(date +%H:%M:%S)] DONE — omnimergekit + modelopt + vllm envs ready"
POD
