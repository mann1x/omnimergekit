#!/bin/bash
# Launch the isolate-Term_Pref router trainer on GPU0, log to t174 dir.
set -uo pipefail
mkdir -p /srv/ml/logs/t174
cd /srv/ml/scripts
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error \
  nohup /srv/ml/envs/envs/omnimergekit/bin/python /srv/ml/scripts/router_term_pref.py "$@" \
  > /srv/ml/logs/t174/router_tp_train.log 2>&1 &
echo "ROUTER_TP_PID $!"
