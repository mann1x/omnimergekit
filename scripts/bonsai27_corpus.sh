#!/bin/bash
# Omnimerge v6 (Qwen3.8-27B) replay corpus: plan -> one vLLM server on GPU1 -> generate STAGE (eval | train) rows.
# RUN names the output dir; a small PLAN_N with STAGE=train is the spot check. Resumable (ids already written are skipped).
set -u
W=/mnt/sdc/opencoti-scratch/bonsai27/${RUN:?RUN must be set}; M=/mnt/sdc/opencoti-models/hf/ManniX-ITA/Qwen3.8-27B-Omnimerge-v6
D=/mnt/sdc/opencoti-models/bonsai17/data; L=$W/corpus.log; PY=/root/anaconda3/envs/omnimergekit/bin/python
STAGES=${STAGES:?STAGES must be set, e.g. "eval train"}
mkdir -p $W; ts() { date -u +%H:%M:%SZ; }
if [ ! -s $W/plan_train.jsonl ]; then
  $PY /srv/ml/scripts/bonsai_corpus_plan.py --corpus /mnt/sdc/ml/omnimerge-v6/calib_train.txt --code $D/codefeedback/codefeedback.jsonl \
    --general $D/dolly/databricks-dolly-15k.jsonl --out $W --n "${PLAN_N:?PLAN_N must be set}" --samplers "${PLAN_SAMPLERS:?PLAN_SAMPLERS must be set}" \
    --efforts "${PLAN_EFFORTS:?PLAN_EFFORTS must be set}" --n-eval ${N_EVAL:-48} > $W/plan.out 2>&1 || { echo "[$(ts)] FATAL plan" >> $L; exit 1; }
  echo "[$(ts)] plan: $(wc -l < $W/plan_train.jsonl) train, $(wc -l < $W/plan_eval.jsonl) eval" >> $L
fi
CUDA_VISIBLE_DEVICES=1 /root/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 8142 --model $M \
  --served-model-name omnimerge-v6-teacher --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization ${VLLM_UTIL:-0.93} \
  --max-num-seqs ${MAX_SEQS:-96} --max-num-batched-tokens 8192 > $W/vllm.log 2>&1 &
SRV=$!; n=0
until curl -sf http://127.0.0.1:8142/v1/models > /dev/null; do
  kill -0 $SRV 2>/dev/null || { echo "[$(ts)] FATAL server died: $(grep -E 'Error' $W/vllm.log | tail -1 | cut -c1-300)" >> $L; exit 1; }
  n=$((n+1)); [ $n -gt 360 ] && { echo "[$(ts)] FATAL server not ready" >> $L; kill $SRV; exit 1; }; sleep 5
done
echo "[$(ts)] server ready; $(grep -oE 'Maximum concurrency[^,]*' $W/vllm.log | tail -1)" >> $L
for st in $STAGES; do
  $PY -u /srv/ml/scripts/bonsai_gen_corpus_srv2.py --url http://127.0.0.1:8142 --expect-model Omnimerge-v6 --plan $W/plan_$st.jsonl \
    --workers ${WORKERS:-64} --retries ${RETRIES:-1} --limit ${LIMIT:-0} --out $W/gen_$st > $W/gen_$st.out 2>&1
  echo "[$(ts)] stage $st: $(tail -1 $W/gen_$st.out | cut -c1-300)" >> $L
done
kill $SRV; wait $SRV 2>/dev/null
echo "[$(ts)] === CORPUS27_DONE ===" >> $L
