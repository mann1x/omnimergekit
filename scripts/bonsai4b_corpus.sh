#!/bin/bash
# Qwen3.5-4B replay corpus: plan (hold-outs fixed first) -> vLLM teacher -> eval traces -> train traces.
set -u
W=/mnt/sdc/opencoti-scratch/bonsai4b/${RUN:?RUN must be set}; M=/mnt/sdc/opencoti-models/bonsai4b/Qwen3.5-4B; L=$W/corpus.log
D=/mnt/sdc/opencoti-models/bonsai17/data; PY=/root/anaconda3/envs/omnimergekit/bin/python
ENVP=${VLLM_ENV:?VLLM_ENV must be set}
ts() { date -u +%H:%M:%SZ; }
mkdir -p $W
if [ ! -s $W/plan_train.jsonl ]; then
  $PY /srv/ml/scripts/bonsai_corpus_plan.py --corpus /mnt/sdc/ml/omnimerge-v6/calib_train.txt --code $D/codefeedback/codefeedback.jsonl \
    --general $D/dolly/databricks-dolly-15k.jsonl --out $W --n "${PLAN_N:?PLAN_N must be set}" --samplers "${PLAN_SAMPLERS:?PLAN_SAMPLERS must be set}" \
    > $W/plan.out 2>&1 || { echo "[$(ts)] FATAL plan failed" >> $L; exit 1; }
fi
echo "[$(ts)] plan: $(wc -l < $W/plan_train.jsonl) train, $(wc -l < $W/plan_eval.jsonl) eval prompts" >> $L
CUDA_VISIBLE_DEVICES=1 $ENVP/bin/python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 8142 --model $M \
  --served-model-name qwen3.5-4b-teacher --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization ${VLLM_UTIL:?VLLM_UTIL must be set (cap on TOTAL card usage)} \
  --max-num-seqs 256 --max-num-batched-tokens 8192 > $W/vllm.log 2>&1 &
SRV=$!
echo "[$(ts)] vllm ($ENVP) pid $SRV" >> $L
until curl -sf http://127.0.0.1:8142/v1/models > /dev/null; do
  kill -0 $SRV 2>/dev/null || { echo "[$(ts)] FATAL server died: $(grep -E 'Error|error' $W/vllm.log | tail -2 | cut -c1-300)" >> $L; exit 1; }
  sleep 5
done
echo "[$(ts)] server ready" >> $L
for part in eval train; do
  $PY -u /srv/ml/scripts/bonsai_gen_corpus_srv.py --url http://127.0.0.1:8142 --expect-model Qwen3.5-4B --plan $W/plan_$part.jsonl --out $W/gen_$part >> $W/gen_$part.out 2>&1
  echo "[$(ts)] $part: $(tail -1 $W/gen_$part.out | cut -c1-600)" >> $L
done
kill $SRV; wait $SRV 2>/dev/null
echo "[$(ts)] === CORPUS_DONE ===" >> $L
