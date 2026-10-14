#!/bin/bash
# 1.7B: teacher self-traces, then the "is a trace mix supported?" test:
# score PrismML, alllwang and our two 8k checkpoints on held-out teacher traces next to wiki.
set -u
W=/mnt/sdc/opencoti-scratch/bonsai17; M=/mnt/sdc/opencoti-models/bonsai17; T=$W/traces
PY=/root/anaconda3/envs/omnimergekit/bin/python; Q=/srv/ml/scripts/bonsai17_qat.v3.py
WIKI=/srv/ml/ppl-corpus/wiki.test.mid.raw; L=$W/r4_support.log
ts() { date -u +%H:%M:%SZ; }
mkdir -p "$T"
until grep -qE "R3_CHAIN_DONE|FATAL" $W/r3_chain.log; do sleep 20; done
echo "[$(ts)] r3 chain finished, generating traces" >> $L
$PY -u /srv/ml/scripts/bonsai17_gen_traces.py --base $M/Qwen3-1.7B --corpus /mnt/sdc/ml/omnimerge-v6/calib_train.txt \
  --out-train $T/train_traces.txt --out-eval $T/heldout_traces.txt --heldout 64 --samples 3 --bs 128 > $W/r4_gen.out 2>&1 &
GEN=$!
until grep -q "heldout_traces.txt\", \"convs\"" $W/r4_gen.out; do
  kill -0 $GEN 2>/dev/null || { echo "[$(ts)] FATAL generator died" >> $L; exit 1; }
  sleep 10
done
echo "[$(ts)] $(grep STATS $W/r4_gen.out | head -1)" >> $L
E="--base $M/Qwen3-1.7B --eval-text $WIKI --eval-text-b $T/heldout_traces.txt"
for ck in prism-unpacked alllwang; do
  $PY -u $Q $E --eval-only $M/$ck > $W/r4_eval_$ck.out 2>&1
  echo "[$(ts)] $ck: $(grep EVAL $W/r4_eval_$ck.out)" >> $L
done
for arm in top86 gptq; do
  $PY -u $Q $E --init $arm --train-parquet $M/data/fineweb-edu-10BT-000.parquet --load $W/r2_${arm}_8k/latent.pt \
    --steps 0 --out $W/r4_eval_${arm}_8k > $W/r4_eval_${arm}_8k.out 2>&1
  echo "[$(ts)] ours ${arm} 8k: $(grep EVAL $W/r4_eval_${arm}_8k.out | tail -1)" >> $L
done
echo "[$(ts)] === R4_SUPPORT_DONE ===" >> $L
wait $GEN
echo "[$(ts)] $(grep STATS $W/r4_gen.out | tail -1)" >> $L
echo "[$(ts)] === R4_TRACES_DONE ===" >> $L
