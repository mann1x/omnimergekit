#!/bin/bash
# Self-knowledge probe (scores, parameters, training data, version): Bonsai 2 on the PrismML engine, then base
# Qwen3.8-27B bf16 on vLLM. Waits for the first base probe to release the GPU.
set -u
PY=/root/anaconda3/envs/omnimergekit/bin/python; PRISM=/srv/ml/bonsai-assess/ref-b10754/llama-prism-b10754-2459f68
W=/mnt/sdc/opencoti-scratch/v6-ternary/identity_probe; L=$W/probe2.log
until grep -qE "IDENTITY_PROBE_BASE_DONE|FATAL vllm" $W/probe.log; do sleep 5; done
LD_LIBRARY_PATH="$PRISM" CUDA_VISIBLE_DEVICES=1 "$PRISM/llama-server" -m /mnt/sdc/opencoti-models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf --host 127.0.0.1 --port 8131 -ngl 99 -c 32768 -np 4 --no-warmup > $W/server2.bonsai2.log 2>&1 &
SP=$!; i=0
until curl -sf http://127.0.0.1:8131/health 2>/dev/null | grep -q '"ok"'; do i=$((i+1)); kill -0 $SP 2>/dev/null || { echo "FATAL bonsai2 server died" >> $L; exit 1; }; [ $i -gt 240 ] && { kill $SP; exit 1; }; sleep 5; done
$PY /srv/ml/scripts/bonsai_identity_probe2.py --url http://127.0.0.1:8131 --tag bonsai2 --out $W/probe2.jsonl >> $L 2>&1
kill $SP; wait $SP 2>/dev/null
CUDA_VISIBLE_DEVICES=1 /root/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 8132 \
  --model /mnt/sdc/opencoti-models/hf/Qwen/Qwen3.8-27B --served-model-name base --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.66 --max-num-seqs 8 --max-num-batched-tokens 8192 > $W/vllm2.base.log 2>&1 &
SP=$!; i=0
until curl -sf http://127.0.0.1:8132/v1/models 2>/dev/null | grep -q '"base"'; do i=$((i+1)); kill -0 $SP 2>/dev/null || { echo "FATAL vllm base died" >> $L; exit 1; }; [ $i -gt 240 ] && { kill $SP; exit 1; }; sleep 5; done
$PY /srv/ml/scripts/bonsai_identity_probe2.py --url http://127.0.0.1:8132 --tag base --out $W/probe2.jsonl >> $L 2>&1
kill $SP; wait $SP 2>/dev/null
echo "=== PROBE2_DONE ===" >> $L
