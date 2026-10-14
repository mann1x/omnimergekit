#!/usr/bin/env bash
# waits for the QAT pilot (literal PID) to release GPU1, then runs the 4-block PT2-style arms
PID=4170157; i=0
while kill -0 $PID 2>/dev/null; do i=$((i+1)); [ $i -ge 480 ] && { echo "FATAL pilot still running after 2h"; exit 1; }; sleep 15; done
echo "pilot gone at $(date -u +%T)Z"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
cd /srv/ml/scripts; LAB=/mnt/sdc/opencoti-scratch/v6-ternary/lab
/root/anaconda3/envs/omnimergekit/bin/python -u bonsai_pt2_blocks.py --model-dir /mnt/sdc/opencoti-models/hf/Qwen/Qwen3.8-27B \
  --prism-gguf /mnt/sdc/opencoti-models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf --pack $LAB/pack_blk0-3.pt \
  --calib-text /mnt/sdc/ml/omnimerge-v6/calib_train.txt --eval-text /srv/ml/ppl-corpus/wiki.test.mid.raw \
  --out $LAB/pt2_blocks_a.json --device cuda:1
