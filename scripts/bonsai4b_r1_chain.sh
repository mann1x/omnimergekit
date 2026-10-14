#!/bin/bash
# Qwen3.5-4B round 1: 25% trace mix (family-weighted) then the FineWeb-only control. 2,000 steps each, strict top-86 format.
set -u
W=/mnt/sdc/opencoti-scratch/bonsai4b; C=$W/corpus_v1; L=$W/r1_chain.log
ts() { date -u +%H:%M:%SZ; }
run() {  # name, extra args...
  local name=$1; shift
  mkdir -p $W/$name
  echo "[$(ts)] start $name" >> $L
  PYTHONPATH=/srv/ml/pylibs/fla PYTORCH_ALLOC_CONF=expandable_segments:True /root/anaconda3/envs/omnimergekit/bin/python -u /srv/ml/scripts/bonsai4b_qat.py \
    --base /mnt/sdc/opencoti-models/bonsai4b/Qwen3.5-4B --train-parquet /mnt/sdc/opencoti-models/bonsai17/data/fineweb-edu-10BT-000.parquet \
    --eval-text /srv/ml/ppl-corpus/wiki.test.mid.raw --eval-text-b $C/eval_code.jsonl,$C/eval_tech.jsonl,$C/eval_general.jsonl,$C/agent_eval.jsonl \
    --init top86 --bs 2 --accum 4 --steps 2000 --eval-every 250 --save-every 1000 --out $W/$name --device cuda:1 "$@" > $W/$name/run.out 2>&1
  echo "[$(ts)] $name rc=$? $(grep -E '^EVAL' $W/$name/run.out | tail -1 | cut -c1-420)" >> $L
}
run r1_mix25 --mix-text $C/train_code.jsonl=0.4,$C/train_tech.jsonl=0.2,$C/train_general.jsonl=0.25,$C/agent_train.jsonl=0.15 --mix-frac 0.25
run r1_ctrl
echo "[$(ts)] === R1_CHAIN_DONE ===" >> $L
