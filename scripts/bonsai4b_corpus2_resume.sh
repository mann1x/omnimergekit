#!/bin/bash
# Resume the train stage against the three already-running servers with more streams per server; stop the servers at the end.
set -u
W=/mnt/sdc/opencoti-scratch/bonsai4b/corpus_v1; L=$W/corpus.log; PY=/root/anaconda3/envs/omnimergekit/bin/python
ts() { date -u +%H:%M:%SZ; }
DONE=$(ls $W/gen_train*.jsonl)
GENS=()
for i in 0 1 2; do
  $PY -u /srv/ml/scripts/bonsai_gen_corpus_srv2.py --url http://127.0.0.1:$((8142+i)) --expect-model Qwen3.5-4B --plan $W/plan_train.jsonl \
    --shard $i/3 --workers 140 --done-from $DONE --out $W/gen_train.r$i >> $W/gen_train.r$i.out 2>&1 &
  GENS+=($!)
done
echo "[$(ts)] resume generators ${GENS[@]} (140 streams per server)" >> $L
wait ${GENS[@]}
for i in 0 1 2; do echo "[$(ts)] shard $i: $(tail -1 $W/gen_train.r$i.out | cut -c1-600)" >> $L; done
kill 2629268 2629936 2630549
echo "[$(ts)] === CORPUS2_DONE ===" >> $L
