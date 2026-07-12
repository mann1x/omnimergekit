#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/srv/ml/envs/envs/omk-yarn/bin/python ; PROBE=/srv/ml/scripts/mem_probe_longctx.py
MOE=/srv/ml/google/gemma-4-26B-A4B-it ; B31=/mnt/sdc/ml/google/gemma-4-31B-it
# MoE ceiling on GPU1
for S in 65536 131072 196608; do
  echo "### MoE seqlen=$S ###"
  $PY $PROBE --model-dir $MOE --seqlen $S --steps 1 --offload-activations --attn memeff --ce-chunk 1024 --gpu 1 2>&1 | grep -E "step 0|RESULT|OutOfMemory|Tried to allocate" | tail -2
done
# 31B ceiling on GPU0
for S in 65536 98304; do
  echo "### 31B seqlen=$S ###"
  $PY $PROBE --model-dir $B31 --seqlen $S --steps 1 --offload-activations --attn memeff --ce-chunk 1024 --gpu 0 2>&1 | grep -E "step 0|RESULT|OutOfMemory|Tried to allocate" | tail -2
done
echo "### CEILING SWEEP DONE ###"
