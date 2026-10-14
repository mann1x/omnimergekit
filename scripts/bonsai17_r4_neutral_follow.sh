#!/bin/bash
# Score the control and the 50% run on the neutral set as each finishes.
W=/mnt/sdc/opencoti-scratch/bonsai17
until grep -qE "r4_ctrl:|FATAL" $W/r4_mix_chain.log; do sleep 20; done
bash /srv/ml/scripts/bonsai17_r4_neutral.sh r4_ctrl
until grep -qE "R4_MIX_CHAIN_DONE|FATAL" $W/r4_mix_chain.log; do sleep 20; done
bash /srv/ml/scripts/bonsai17_r4_neutral.sh r4_mix50
echo "=== R4_ALL_DONE ===" >> $W/r4_neutral.log
