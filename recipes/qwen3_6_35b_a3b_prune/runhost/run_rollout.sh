#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=1
PY=/root/anaconda3/envs/omnimergekit/bin/python
W=/mnt/sdc/ream-work
# wait for the P(EOG) chain to release GPU1 -- never two jobs on one GPU
while pgrep -f "[e]og_prob_probe.py" >/dev/null 2>&1; do sleep 30; done
echo "ROLLOUT: GPU1 released, starting $(date -u +%H:%M:%S)Z"
run() { echo "=== $1 ==="; "$PY" "$W/empty_rollout_probe.py" --label "$1" --model "$2" \
        --cell "$3" --out "$W/rollout_$1.json" 2>&1 | grep -vE "it/s\]|Loading weights"; }
run armI "$W/armI"                 hybrid_p12_ourssal_reapfloor
run armJ "$W/armJ"                 hybrid_p24_ourssal_reapfloor
run armD "$W/armD_ourssal_nomerge" reamD_ourssal_nomerge
echo "=== ROLLOUT_CHAIN_DONE ==="
