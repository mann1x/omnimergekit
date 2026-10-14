#!/bin/bash
# 1.7B round 5, steps 2 + 3: second neutral eval (GSM8K test questions, complete teacher conversations), reference models
# re-scored on it, then the 25% mix rerun on the complete-conversation trace set. One GPU job at a time.
set -u
W=/mnt/sdc/opencoti-scratch/bonsai17; M=/mnt/sdc/opencoti-models/bonsai17; T=$W/traces; L=$W/r5_chain.log
PY=/root/anaconda3/envs/omnimergekit/bin/python; Q=/srv/ml/scripts/bonsai17_qat.v4.py
WIKI=/srv/ml/ppl-corpus/wiki.test.mid.raw; CORPUS=/mnt/sdc/ml/omnimerge-v6/calib_train.txt
ts() { date -u +%H:%M:%SZ; }
D=$M/data/gsm8k; mkdir -p $D
[ -s $D/test.parquet ] || aria2c -q -x4 -d $D -o test.parquet https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet
[ -s $D/test.parquet ] || { echo "[$(ts)] FATAL gsm8k download" >> $L; exit 1; }
$PY - >> $L 2>&1 <<PY
import json, pyarrow.parquet as pq
corpus = open("$CORPUS").read() + open("$T/train_full.clean.jsonl").read()
qs = pq.read_table("$D/test.parquet").column("question").to_pylist()
keep = [q for q in qs if q.strip() not in corpus and json.dumps(q.strip())[1:-1] not in corpus]
open("$D/test.jsonl", "w").write("".join(json.dumps({"instruction": q, "context": ""}) + "\n" for q in keep))
print(f"gsm8k test: {len(qs)} questions, {len(qs) - len(keep)} found in calibration corpus or training traces, {len(keep)} usable")
PY
if [ ! -s $T/neutral_b.jsonl ]; then
  CUDA_VISIBLE_DEVICES=1 /root/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 --port 8141 --model $M/Qwen3-1.7B --served-model-name qwen3-1.7b-teacher --dtype bfloat16 \
    --max-model-len 16384 --gpu-memory-utilization 0.5 --max-num-seqs 256 > $W/r5_vllm_b.log 2>&1 &
  SRV=$!
  until curl -sf http://127.0.0.1:8141/v1/models > /dev/null; do
    kill -0 $SRV 2>/dev/null || { echo "[$(ts)] FATAL server died" >> $L; exit 1; }
    sleep 5
  done
  $PY -u /srv/ml/scripts/bonsai17_gen_traces_srv.py --expect-model Qwen3-1.7B --jsonl $D/test.jsonl --jsonl-name gsm8k \
    --out $T/neutral_b --limit 160 --max-new 4096 > $W/r5_neutral_b_gen.out 2>&1
  kill $SRV; wait $SRV 2>/dev/null
  echo "[$(ts)] neutral_b: $(tail -1 $W/r5_neutral_b_gen.out | cut -c1-200) sha $(sha256sum $T/neutral_b.jsonl | cut -c1-16)" >> $L
fi
[ -s $T/neutral_b.jsonl ] || { echo "[$(ts)] FATAL no neutral_b" >> $L; exit 1; }
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -lt 2000 ]; do sleep 3; done
E="--base $M/Qwen3-1.7B --eval-text $WIKI --eval-text-b $T/neutral_b.jsonl"
for ck in prism-unpacked alllwang; do
  $PY -u $Q $E --eval-only $M/$ck > $W/r5_nb_$ck.out 2>&1
  echo "[$(ts)] $ck: $(grep -E 'EVAL|Error' $W/r5_nb_$ck.out | tail -1 | cut -c1-260)" >> $L
done
echo "[$(ts)] $(grep 'teacher ppl' $W/r5_nb_alllwang.out | tail -1)" >> $L
for run in r2_top86_8k r4_ctrl r4_mix25 r4_mix50; do
  $PY -u $Q $E --init top86 --train-parquet $M/data/fineweb-edu-10BT-000.parquet --load $W/$run/latent.pt --steps 0 --out $W/r5_nb_$run > $W/r5_nb_$run.out 2>&1
  echo "[$(ts)] $run: $(grep -E 'EVAL|Error' $W/r5_nb_$run.out | tail -1 | cut -c1-260)" >> $L
done
echo "[$(ts)] === R5_NEUTRAL_B_DONE ===" >> $L
$PY -u $Q --base $M/Qwen3-1.7B --train-parquet $M/data/fineweb-edu-10BT-000.parquet \
  --eval-text $WIKI --eval-text-b $T/heldout_traces.txt,$T/neutral_traces.txt,$T/neutral_b.jsonl \
  --init top86 --steps 2000 --save-every 2000 --mix-text $T/train_full.clean.jsonl --mix-frac 0.25 --out $W/r5_mix25 > $W/r5_mix25.out 2>&1
echo "[$(ts)] r5_mix25: $(grep -E 'EVAL|Error' $W/r5_mix25.out | tail -1 | cut -c1-520)" >> $L
echo "[$(ts)] === R5_CHAIN_DONE ===" >> $L
