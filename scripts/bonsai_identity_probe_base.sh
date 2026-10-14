#!/bin/bash
# Identity probe, base Qwen3.8-27B bf16 served by vLLM (the bf16 GGUF is no longer on disk).
set -u
W=/mnt/sdc/opencoti-scratch/v6-ternary/identity_probe
CUDA_VISIBLE_DEVICES=1 /root/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 8132 \
  --model /mnt/sdc/opencoti-models/hf/Qwen/Qwen3.8-27B --served-model-name base --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.66 --max-num-seqs 8 --max-num-batched-tokens 8192 > $W/vllm.base.log 2>&1 &
SP=$!; i=0
until curl -sf http://127.0.0.1:8132/v1/models 2>/dev/null | grep -q '"base"'; do
  i=$((i+1)); kill -0 $SP 2>/dev/null || { echo "FATAL vllm base died: $(grep Error $W/vllm.base.log | tail -1 | cut -c1-240)" >> $W/probe.log; exit 1; }
  [ $i -gt 240 ] && { echo "FATAL vllm base not ready" >> $W/probe.log; kill $SP; exit 1; }; sleep 5
done
/root/anaconda3/envs/omnimergekit/bin/python /srv/ml/scripts/bonsai_identity_probe.py --url http://127.0.0.1:8132 --tag base --out $W/probe.jsonl >> $W/probe.log 2>&1
kill $SP; wait $SP 2>/dev/null
echo "=== IDENTITY_PROBE_BASE_DONE ===" >> $W/probe.log
