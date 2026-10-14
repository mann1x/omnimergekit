#!/bin/bash
# Qwen3.5-4B replay corpus, train stage on N vLLM servers sharing GPU1 (the engine is CPU-bound, so servers add up).
# Resumes: ids already in any gen_train*.jsonl are skipped.
set -u
W=/mnt/sdc/opencoti-scratch/bonsai4b/${RUN:?RUN must be set}; M=/mnt/sdc/opencoti-models/bonsai4b/Qwen3.5-4B; L=$W/corpus.log
PY=/root/anaconda3/envs/omnimergekit/bin/python; ENVP=${VLLM_ENV:?VLLM_ENV must be set}
N=${NSRV:?NSRV must be set}; CAPS=(${VLLM_CAPS:?VLLM_CAPS must be set: cumulative caps on TOTAL card usage, one per server})
ts() { date -u +%H:%M:%SZ; }
[ -s $W/plan_train.jsonl ] || { echo "[$(ts)] FATAL no plan" >> $L; exit 1; }
SRVS=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=1 $ENVP/bin/python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port $((8142+i)) --model $M \
    --served-model-name qwen3.5-4b-teacher --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization ${CAPS[$i]} \
    --max-num-seqs 256 --max-num-batched-tokens 8192 > $W/vllm.s$i.log 2>&1 &
  SRVS+=($!)
  until curl -sf http://127.0.0.1:$((8142+i))/v1/models > /dev/null; do
    kill -0 ${SRVS[$i]} 2>/dev/null || { echo "[$(ts)] FATAL server $i died: $(grep -E 'Error|error' $W/vllm.s$i.log | tail -2 | cut -c1-300)" >> $L; kill ${SRVS[@]} 2>/dev/null; exit 1; }
    sleep 5
  done
  echo "[$(ts)] server $i ready (pid ${SRVS[$i]}, cap ${CAPS[$i]})" >> $L
done
GENS=()
for i in $(seq 0 $((N-1))); do
  $PY -u /srv/ml/scripts/bonsai_gen_corpus_srv2.py --url http://127.0.0.1:$((8142+i)) --expect-model Qwen3.5-4B --plan $W/plan_train.jsonl \
    --shard $i/$N --workers ${WORKERS:-144} --done-from $(ls $W/gen_train*.jsonl) --out $W/gen_train.s$i >> $W/gen_train.s$i.out 2>&1 &
  GENS+=($!)
done
echo "[$(ts)] generators ${GENS[@]}" >> $L
wait ${GENS[@]}
for i in $(seq 0 $((N-1))); do echo "[$(ts)] shard $i: $(tail -1 $W/gen_train.s$i.out | cut -c1-600)" >> $L; done
kill ${SRVS[@]}; wait 2>/dev/null
echo "[$(ts)] === CORPUS2_DONE ===" >> $L
