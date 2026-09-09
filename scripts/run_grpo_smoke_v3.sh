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

# --liger IS DELIBERATELY ABSENT. It used to be here, and it was WRONG.
#
# It was added to dodge an OOM in selective_log_softmax. `vllm_enable_sleep_mode` has
# since removed that OOM at the source by releasing vLLM's weights AND kv_cache during
# the training step, so Liger buys nothing -- the no-Liger control below fits with room
# to spare.
#
# Worse, Liger's fused GRPO loss was CORRUPTING the run. Controlled A/B on 2026-09-09,
# identical code/data/seed, only this flag differing, measured with the in-run NANPROBE
# callback:
#
#            grad_norm   kl       loss        step
#   +liger   nan         0.3166   0.02657     226.5 s   grads nonfinite 410/410
#   -liger   0.02768     0        0.00598     255.6 s   grads nonfinite   0/410
#
# The nan entered at microbatch 3 of 31 and, because gradients accumulate, poisoned
# every later microbatch -- so `grad_norm nan` looked constant and structural when it was
# actually data-dependent. `kl` is the tell: at step 1 LoRA has B=0 and lora_dropout=0,
# so the policy and the adapter-disabled reference are the SAME FUNCTION and the KL is
# exactly 0. Liger reported 0.3166. It was not measuring the model.
#
# That one flag also produced the 1.391e7 loss spike and the apparent policy divergence
# in the 4-step smoke. None of it was real: PARAMS nonfinite stayed 0/410 throughout,
# which is why the tiers never collapsed.
#
# Cost of not using it: ~13% step time. Do NOT re-add it without re-running that A/B.
# And do NOT reach for PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True if an OOM ever
# returns -- it uses the CUDA VMM API, breaks cudaIpcGetMemHandle, and kills vLLM TP=2.

exec /srv/ml/envs/envs/omk-grpo/bin/torchrun --nproc_per_node 2 --standalone \
  scripts/train_grpo_efficiency.py \
  --model /mnt/sdc/v7rework/arms/Jprime-p3-bf16 \
  --pool eval/efficiency/grpo_pool_v3_presmoke.jsonl \
  --output /mnt/sdc/v7rework/grpo_smoke_v3 \
  --smoke --smoke-rows-per-tier 6 --smoke-steps 4 \
  --max-completion-len 8192 \
  --measure-out /mnt/sdc/v7rework/grpo_smoke_v3/measured.json
