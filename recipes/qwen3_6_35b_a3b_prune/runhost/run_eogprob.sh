#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=1
PY=/root/anaconda3/envs/omnimergekit/bin/python
W=/mnt/sdc/ream-work
run() {  # label  modeldir  cell
  echo "=== $1 ==="
  "$PY" "$W/eog_prob_probe.py" --label "$1" --model "$2" --cell "$3" \
        --out "$W/eogprob_$1.json" 2>&1 | grep -vE "it/s\]|Loading weights"
}
run armD "$W/armD_ourssal_nomerge" reamD_ourssal_nomerge
run armI "$W/armI"                 hybrid_p12_ourssal_reapfloor
run armJ "$W/armJ"                 hybrid_p24_ourssal_reapfloor
run base256e /srv/ml/models/Qwen3.6-35B-A3B base256e_imat
echo "=== EOGPROB_CHAIN_DONE ==="
