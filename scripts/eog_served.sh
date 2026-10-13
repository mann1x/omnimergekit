#!/usr/bin/env bash
# EOG agreement, SERVED rendering regime (preserve_thinking off) on TRIMMED traces.
# Supersedes every earlier run: those used the old prefix-render close detector
# (bug-370) AND the Tier-B regime preserve_thinking=True, which is what the template
# comment blames for the v7-coder loop. No continuation pass here: generate() needs a
# KV cache and 16k of it does not fit beside 52 GiB of bf16 weights.
set -u
cd /mnt/sdc/harness
CUDA_VISIBLE_DEVICES=0 /srv/ml/envs/envs/omnimergekit/bin/python3.11 eog_agreement_probe.py \
  --self-control --require-control --ctx 16384 \
  --traces v4_qwen=/mnt/sdc/agentbench/tierb/manic_v4_fixed_71_trim.json \
  --traces gemma4_fam=/mnt/sdc/agentbench/tierb/gemma4_control_11_trim.json \
  --out /mnt/sdc/harness/eog_rows_served_16k.json
echo "EOG_SERVED_DONE rc=$?"
