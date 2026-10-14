#!/bin/bash
# Spot check of replay-corpus generation on Qwen3.5-4B (thinking on / effort sentence removed / thinking off).
set -u
W=/mnt/sdc/opencoti-scratch/bonsai4b; M=/mnt/sdc/opencoti-models/bonsai4b/Qwen3.5-4B; L=$W/spotcheck.log
ENVP=${VLLM_ENV:?VLLM_ENV must be set}
ts() { date -u +%H:%M:%SZ; }
mkdir -p $W
CUDA_VISIBLE_DEVICES=1 $ENVP/bin/python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 8142 --model $M \
  --served-model-name qwen3.5-4b-teacher --dtype bfloat16 --max-model-len ${MAXLEN:-16384} --gpu-memory-utilization ${VLLM_UTIL:?VLLM_UTIL must be set (cap on TOTAL card usage, other processes included)} \
  --max-num-seqs 128 --max-num-batched-tokens 8192 > $W/vllm.log 2>&1 &
SRV=$!
echo "[$(ts)] vllm ($ENVP) pid $SRV" >> $L
until curl -sf http://127.0.0.1:8142/v1/models > /dev/null; do
  kill -0 $SRV 2>/dev/null || { echo "[$(ts)] FATAL server died: $(grep -E 'Error|error' $W/vllm.log | tail -2 | cut -c1-300)" >> $L; exit 1; }
  sleep 5
done
echo "[$(ts)] server ready" >> $L
/root/anaconda3/envs/omnimergekit/bin/python -u /srv/ml/scripts/bonsai_trace_spotcheck.py --url http://127.0.0.1:8142 \
  --expect-model Qwen3.5-4B --corpus /mnt/sdc/ml/omnimerge-v6/calib_train.txt \
  --jsonl /mnt/sdc/opencoti-models/bonsai17/data/dolly/databricks-dolly-15k.jsonl --out $W/${OUT:-spot_4b} ${SPOT_ARGS:-} > $W/${OUT:-spot_4b}.out 2>&1
grep -E "ROW|server|Traceback|Error" $W/${OUT:-spot_4b}.out >> $L
kill $SRV; wait $SRV 2>/dev/null
echo "[$(ts)] === SPOTCHECK_DONE ===" >> $L
