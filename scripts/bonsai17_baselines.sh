#!/usr/bin/env bash
# baselines in one harness: PrismML 1.7B, the community reproduction, and both starts at step 0 (+3-step smoke)
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python; B=/mnt/sdc/opencoti-models/bonsai17; W=/mnt/sdc/opencoti-scratch/bonsai17
C="--base $B/Qwen3-1.7B --eval-text /srv/ml/ppl-corpus/wiki.test.mid.raw --device cuda:1"
cd /srv/ml/scripts
for m in prism-unpacked alllwang; do $PY -u bonsai17_qat.py $C --eval-only $B/$m 2>&1 | grep -E "EVAL|eval:|Error|Traceback" ; done
for i in top86 gptq; do $PY -u bonsai17_qat.py $C --train-parquet $B/data/fineweb-edu-10BT-000.parquet --init $i --steps 3 --eval-every 3 --save-every 100000 --out $W/smoke_$i 2>&1 | grep -vE "Warning|warn" | tail -12; done
echo "=== B17_BASE_DONE ==="
