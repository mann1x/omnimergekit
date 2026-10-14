#!/usr/bin/env bash
# r2: calibrated start, learning-rate check at 2000 steps (2e-4 vs the 6e-5 of r1), then 8000 steps at the better one.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python; B=/mnt/sdc/opencoti-models/bonsai17; W=/mnt/sdc/opencoti-scratch/bonsai17
run(){ # name, lr, steps
  cd /srv/ml/scripts && $PY -u bonsai17_qat.py --base $B/Qwen3-1.7B --eval-text /srv/ml/ppl-corpus/wiki.test.mid.raw --device cuda:1 \
     --train-parquet $B/data/fineweb-edu-10BT-000.parquet --init gptq --lr "$2" --steps "$3" --eval-every 500 --save-every 2000 --out "$W/$1" > "$W/$1.out" 2>&1
  grep -q QAT17_DONE "$W/$1.out" || { echo "FATAL $1"; tail -4 "$W/$1.out"; exit 1; }
}
kl(){ $PY -c "import json,sys;print(json.load(open(sys.argv[1]))['results'][-1]['kl'])" "$W/$1/summary.json"; }
run r2_gptq_lr2e-4 2e-4 2000
A=$(kl r1_gptq); C=$(kl r2_gptq_lr2e-4); echo "final KL @2000: lr 6e-5 = $A   lr 2e-4 = $C"
LR=$($PY -c "import sys;print('2e-4' if float(sys.argv[2])<float(sys.argv[1]) else '6e-5')" "$A" "$C"); echo "long run at lr $LR"
run r2_gptq_8k "$LR" 8000
echo "=== R2_CHAIN_DONE ==="
