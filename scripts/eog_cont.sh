#!/usr/bin/env bash
# Continuation test, served regime, trimmed traces.
#
# 16k OOM'd on the generate path: a single forward runs with use_cache=False, but
# generate() must build a KV cache, and 30 layers of it at 16k does not fit beside
# 52 GiB of bf16 weights. The failure is a FRAGMENTATION one -- 5.01 GiB reserved but
# unallocated against a 4.00 GiB request -- which is the one case expandable_segments
# actually addresses. The project ban on that flag is SCOPED to CUDA-IPC contexts
# (vLLM TP>1, colocate) because it uses the CUDA VMM API and breaks
# cudaIpcGetMemHandle; this is a single-process probe, so the ban does not reach it.
# torch 2.10 renamed the var to PYTORCH_ALLOC_CONF.
#
# If it still does not fit, fall back to 8192 rather than leaving the question open.
set -u
cd /mnt/sdc/harness
run() {
  CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  /srv/ml/envs/envs/omnimergekit/bin/python3.11 eog_agreement_probe.py \
    --ctx "$1" --continuation-k 96 \
    --traces v4_qwen=/mnt/sdc/agentbench/tierb/manic_v4_fixed_71_trim.json \
    --traces gemma4_fam=/mnt/sdc/agentbench/tierb/gemma4_control_11_trim.json \
    --out "/mnt/sdc/harness/eog_rows_cont_$1.json"
}
echo ">>> ATTEMPT ctx=16384 with expandable_segments"
if run 16384; then
  echo "EOG_CONT_DONE ctx=16384 rc=0"
else
  echo ">>> 16384 FAILED, falling back to ctx=8192 (coverage drops; report it)"
  run 8192 && echo "EOG_CONT_DONE ctx=8192 rc=0" || echo "EOG_CONT_DONE rc=1 BOTH FAILED"
fi
