#!/bin/bash
# 1.7B: neutral third eval set. Teacher traces on Dolly-15k instructions (no system prompt, disjoint from our
# calibration corpus), then every model scored on it next to wiki.
set -u
W=/mnt/sdc/opencoti-scratch/bonsai17; M=/mnt/sdc/opencoti-models/bonsai17; T=$W/traces
PY=/root/anaconda3/envs/omnimergekit/bin/python; Q=/srv/ml/scripts/bonsai17_qat.v3.py; L=$W/r4_neutral.log
ts() { date -u +%H:%M:%SZ; }
D=$M/data/dolly
mkdir -p $D
[ -s $D/databricks-dolly-15k.jsonl ] || aria2c -q -x4 -d $D -o databricks-dolly-15k.jsonl \
  https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
[ -s $D/databricks-dolly-15k.jsonl ] || { echo "[$(ts)] FATAL download" >> $L; exit 1; }
echo "[$(ts)] dolly $(wc -l < $D/databricks-dolly-15k.jsonl) rows sha $(sha256sum $D/databricks-dolly-15k.jsonl | cut -c1-16)" >> $L
if [ ! -s $T/neutral_traces.txt ]; then
  $PY -u /srv/ml/scripts/bonsai17_gen_traces.py --base $M/Qwen3-1.7B --corpus /mnt/sdc/ml/omnimerge-v6/calib_train.txt \
    --jsonl $D/databricks-dolly-15k.jsonl --out-eval $T/neutral_traces.txt --heldout 96 --bs 96 > $W/r4_neutral_gen.out 2>&1
  echo "[$(ts)] $(grep -E 'dropped|distinct|STATS' $W/r4_neutral_gen.out | tr '\n' ' ' | cut -c1-400)" >> $L
fi
E="--base $M/Qwen3-1.7B --eval-text /srv/ml/ppl-corpus/wiki.test.mid.raw --eval-text-b $T/neutral_traces.txt"
for ck in prism-unpacked alllwang; do
  $PY -u $Q $E --eval-only $M/$ck > $W/r4_neu_$ck.out 2>&1
  echo "[$(ts)] $ck: $(grep -E 'EVAL|Error' $W/r4_neu_$ck.out | tail -1)" >> $L
done
echo "[$(ts)] $(grep 'teacher ppl' $W/r4_neu_alllwang.out | tail -1)" >> $L
for run in "$@"; do     # our runs: every one is a top86-format latent
  $PY -u $Q $E --init top86 --train-parquet $M/data/fineweb-edu-10BT-000.parquet --load $W/$run/latent.pt \
    --steps 0 --out $W/r4_neu_$run > $W/r4_neu_$run.out 2>&1
  echo "[$(ts)] $run: $(grep -E 'EVAL|Error' $W/r4_neu_$run.out | tail -1)" >> $L
done
echo "[$(ts)] === R4_NEUTRAL_DONE $* ===" >> $L
