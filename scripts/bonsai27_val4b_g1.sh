#!/bin/bash
# Validate bonsai27_qat.py on Qwen3.5-4B against the AdamW run r1_mix25 (same stream, same eval rows, same schedule):
# teacher cache, then probe runs that stop at step 500 of the 2,000-step schedule. Reference at step 500 (full KL):
# wiki 1.308, code 1.244, tech 0.801, general 1.181, agent 1.030.
set -u
W=/mnt/sdc/opencoti-scratch/bonsai4b; C=$W/corpus_v1; B=/mnt/sdc/opencoti-models/bonsai4b/Qwen3.5-4B; K=$W/cache_r1mix; L=$W/val27.log
PY=/root/anaconda3/envs/omnimergekit/bin/python; ts() { date -u +%H:%M:%SZ; }
export CUDA_VISIBLE_DEVICES=1    # only GPU1 is ours; without this the triton kernel opens a context on GPU0 too
export PYTHONPATH=/srv/ml/pylibs/fla:/srv/ml/scripts    # no expandable_segments: with it the latent build faults (bug-766)
if [ ! -f $K/cache.json ]; then
  $PY /srv/ml/scripts/bonsai27_qat.py --phase cache --device cuda:0 --base $B --cache $K --n-seq ${NSEQ:-16000} \
    --train-parquet /mnt/sdc/opencoti-models/bonsai17/data/fineweb-edu-10BT-000.parquet --mix-frac 0.25 \
    --mix-text $C/train_code.jsonl=0.4,$C/train_tech.jsonl=0.2,$C/train_general.jsonl=0.25,$C/agent_train.jsonl=0.15 \
    --eval-text /srv/ml/ppl-corpus/wiki.test.mid.raw --eval-text-b $C/eval_code.jsonl,$C/eval_tech.jsonl,$C/eval_general.jsonl,$C/agent_eval.jsonl \
    > $K.out 2>&1 || { echo "[$(ts)] FATAL cache" >> $L; exit 1; }
  echo "[$(ts)] cache done" >> $L
fi
for ARM in ${ARMS:?ARMS must be set: space list of name:lr[:extra-args]}; do
  name=${ARM%%:*}; rest=${ARM#*:}; lr=${rest%%:*}; extra=""; [ "$rest" != "$lr" ] && extra=${rest#*:}
  mkdir -p $W/$name
  $PY /srv/ml/scripts/bonsai27_qat.py --phase train --device cuda:0 --base $B --cache $K --out $W/$name --steps 2000 --stop-at ${STOP:-500} --lr $lr \
    --eval-every 250 --save-every 100000 --eval-teacher online ${extra//+/ } > $W/$name/run.out 2>&1
  echo "[$(ts)] $name rc=$? $(grep -E '^EVAL' $W/$name/run.out | tail -1 | cut -c1-420)" >> $L
done
echo "[$(ts)] === VAL27_DONE ===" >> $L
