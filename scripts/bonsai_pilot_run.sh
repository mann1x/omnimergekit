#!/usr/bin/env bash
cd /srv/ml/scripts
PY=/root/anaconda3/envs/omnimergekit/bin/python; M=/mnt/sdc/opencoti-models/hf/Qwen/Qwen3.8-27B; LAB=/mnt/sdc/opencoti-scratch/v6-ternary/lab
P=/mnt/sdc/opencoti-models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf
$PY bonsai_prefix_test.py --model-dir $M --base-gguf /mnt/sdc/opencoti-models/sdc-ml/qwen38-base/Qwen3.8-27B-BF16.gguf --prism-gguf $P --text /srv/ml/ppl-corpus/wiki.test.mid.raw --out $LAB/prefix_blk3.json --save-pack $LAB/pack_blk0-3.pt 2>&1 | grep -E "pack saved|PREFIX_DONE|FATAL|Error"
$PY bonsai_qat_pilot.py --model-dir $M --prism-gguf $P --pack $LAB/pack_blk0-3.pt --train-text /mnt/sdc/ml/omnimerge-v6/calib_train.txt --eval-text /srv/ml/ppl-corpus/wiki.test.mid.raw --out $LAB/qat_pilot_a.json 2>&1 | grep -v -iE "pynvml|fast path"
