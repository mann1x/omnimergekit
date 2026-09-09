#!/usr/bin/env bash
# vLLM COLOCATE requires vllm_tensor_parallel_size to divide the TRAINING world size.
# Single-process + TP=2 fails with "tensor_parallel_size (2) must divide world size
# (1) evenly". So the run is launched with 2 processes: DDP training on both GPUs,
# with ONE vLLM engine sharded across the same two GPUs and shared by both ranks.
# That is the requested topology -- model split over 2 GPUs for generation, both GPUs
# training -- and it is the only geometry in which TP=2 is legal here.
#
# Memory per GPU: ~39 GB policy (DDP replica) + ~20-29 GB vLLM shard. The reference
# policy costs nothing extra because LoRA is in use -- TRL takes reference logprobs by
# DISABLING the adapter, not by loading a second copy of the base model.
set -euo pipefail
cd /srv/ml/repos/omnimergekit
export TOKENIZERS_PARALLELISM=false

# vLLM SKIPS its own P2P validation by default (VLLM_SKIP_P2P_CHECK defaults to "1")
# and assumes the custom all-reduce kernel works. These two cards are PHB-connected
# (PCIe host bridge, no NVLink): cudaDeviceEnablePeerAccess succeeds -- torch reports
# can_device_access_peer=True -- but the IPC-handle path the custom kernel uses does
# not, and TP=2 dies with:
#     Failed: Cuda error custom_all_reduce.cuh:164 invalid argument
# Setting the check to 0 makes vLLM actually TEST P2P and disable the custom kernel
# itself when it fails, falling back to NCCL. Measure, do not assume.
export VLLM_SKIP_P2P_CHECK=0

exec /srv/ml/envs/envs/omk-grpo/bin/torchrun --nproc_per_node 2 --standalone \
  scripts/train_grpo_efficiency.py \
  --model /mnt/sdc/v7rework/arms/Jprime-p3-bf16 \
  --pool eval/efficiency/grpo_pool_v3_presmoke.jsonl \
  --output /mnt/sdc/v7rework/grpo_smoke_v3 \
  --smoke --smoke-rows-per-tier 6 --smoke-steps 4 \
  --max-completion-len 8192 \
  --measure-out /mnt/sdc/v7rework/grpo_smoke_v3/measured.json
