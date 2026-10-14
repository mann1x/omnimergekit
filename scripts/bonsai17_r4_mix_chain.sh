#!/bin/bash
# 1.7B: does mixing the teacher's own traces into the fineweb stream help? Strict Bonsai format (top86 start),
# 2000 steps each, both eval domains reported. Control = same recipe, fineweb only. One arm at a time (bug-759).
set -u
W=/mnt/sdc/opencoti-scratch/bonsai17; M=/mnt/sdc/opencoti-models/bonsai17; T=$W/traces
PY=/root/anaconda3/envs/omnimergekit/bin/python; Q=/srv/ml/scripts/bonsai17_qat.v3.py; L=$W/r4_mix_chain.log
ts() { date -u +%H:%M:%SZ; }
until grep -qE "R4_TRACES_DONE|FATAL" $W/r4_support.log; do sleep 20; done
grep -q R4_TRACES_DONE $W/r4_support.log || { echo "[$(ts)] FATAL no traces" >> $L; exit 1; }
run() {  # name, extra args
  local n=$1; shift
  $PY -u $Q --base $M/Qwen3-1.7B --train-parquet $M/data/fineweb-edu-10BT-000.parquet \
    --eval-text /srv/ml/ppl-corpus/wiki.test.mid.raw --eval-text-b $T/heldout_traces.txt \
    --init top86 --steps 2000 --save-every 2000 --out $W/$n "$@" > $W/$n.out 2>&1
  echo "[$(ts)] $n: $(grep EVAL $W/$n.out | tail -1 | cut -c1-330)" >> $L
}
run r4_mix25 --mix-text $T/train_traces.txt --mix-frac 0.25
run r4_ctrl
run r4_mix50 --mix-text $T/train_traces.txt --mix-frac 0.5
echo "[$(ts)] === R4_MIX_CHAIN_DONE ===" >> $L
