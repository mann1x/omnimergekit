#!/usr/bin/env bash
# r3 (2000 steps each, compare against r1_top86 = 0.905 and r1_gptq = 0.759):
#   r3_gptq86      calibrated start under PrismML's own keep-86 quantizer
#   r3_top86_ramp  magnitude start with a soft-to-hard blend over the first 20% of steps
#   r3_gptq86_ramp both
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python; B=/mnt/sdc/opencoti-models/bonsai17; W=/mnt/sdc/opencoti-scratch/bonsai17
i=0; until grep -qE "QAT17_DONE|Traceback" $W/r2_top86_8k.out; do i=$((i+1)); [ $i -ge 480 ] && { echo "FATAL control not done"; exit 1; }; sleep 15; done
i=0; until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -lt 2000 ]; do i=$((i+1)); [ $i -ge 40 ] && { echo "FATAL GPU1 busy"; exit 1; }; sleep 3; done
run(){ n=$1; shift
  cd /srv/ml/scripts && $PY -u bonsai17_qat.v2.py --base $B/Qwen3-1.7B --eval-text /srv/ml/ppl-corpus/wiki.test.mid.raw --device cuda:1 \
     --train-parquet $B/data/fineweb-edu-10BT-000.parquet --steps 2000 --eval-every 250 --save-every 2000 --out "$W/$n" "$@" > "$W/$n.out" 2>&1
  grep -q QAT17_DONE "$W/$n.out" && echo "$n: $(grep EVAL "$W/$n.out" | tail -1 | cut -c1-110)" || { echo "FAILED $n"; tail -3 "$W/$n.out"; }
}
run r3_gptq86 --init gptq86
run r3_top86_ramp --init top86 --lam-ramp 0.2
run r3_gptq86_ramp --init gptq86 --lam-ramp 0.2
echo "=== R3_CHAIN_DONE ==="
