#!/bin/bash
# 1.7B round 5, step 1: complete teacher conversations from a broad prompt pool (our corpus + Dolly-15k),
# served by vLLM on GPU1. Smoke of 64 prompts first, then the full pool. Server is stopped by its own PID.
set -u
W=/mnt/sdc/opencoti-scratch/bonsai17; M=/mnt/sdc/opencoti-models/bonsai17; T=$W/traces; L=$W/r5_traces.log
PY=/root/anaconda3/envs/omnimergekit/bin/python; G=/srv/ml/scripts/bonsai17_gen_traces_srv.py
ts() { date -u +%H:%M:%SZ; }
CUDA_VISIBLE_DEVICES=1 /root/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port 8141 --model $M/Qwen3-1.7B --served-model-name qwen3-1.7b-teacher --dtype bfloat16 \
  --max-model-len 16384 --gpu-memory-utilization 0.85 --max-num-seqs 384 > $W/r5_vllm.log 2>&1 &
SRV=$!
echo "[$(ts)] vllm pid $SRV" >> $L
until curl -sf http://127.0.0.1:8141/v1/models > /dev/null; do
  kill -0 $SRV 2>/dev/null || { echo "[$(ts)] FATAL server died" >> $L; exit 1; }
  sleep 5
done
echo "[$(ts)] server ready" >> $L
ARGS="--expect-model Qwen3-1.7B --corpus /mnt/sdc/ml/omnimerge-v6/calib_train.txt --jsonl $M/data/dolly/databricks-dolly-15k.jsonl --exclude-traces $T/heldout_traces.txt $T/neutral_traces.txt"
$PY -u $G $ARGS --out $T/smoke_full --limit 64 > $W/r5_smoke.out 2>&1
echo "[$(ts)] smoke: $(tail -1 $W/r5_smoke.out | cut -c1-300)" >> $L
echo "[$(ts)] === R5_SMOKE_DONE ===" >> $L
$PY -u $G $ARGS --out $T/train_full > $W/r5_gen.out 2>&1
echo "[$(ts)] full: $(tail -1 $W/r5_gen.out | cut -c1-300)" >> $L
kill $SRV; wait $SRV 2>/dev/null
echo "[$(ts)] === R5_TRACES_DONE ===" >> $L
