#!/bin/bash
# Identity probe: base Qwen3.8-27B (bf16) vs PrismML Ternary Bonsai 2 27B, same engine (PrismML fork), no system prompt.
set -u
PY=/root/anaconda3/envs/omnimergekit/bin/python
PRISM=/srv/ml/bonsai-assess/ref-b10754/llama-prism-b10754-2459f68
W=/mnt/sdc/opencoti-scratch/v6-ternary/identity_probe; mkdir -p $W
for A in "bonsai2:/mnt/sdc/opencoti-models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf" "base:/mnt/sdc/opencoti-models/sdc-ml/qwen38-base/Qwen3.8-27B-BF16.gguf"; do
  tag=${A%%:*}; gg=${A#*:}
  LD_LIBRARY_PATH="$PRISM" CUDA_VISIBLE_DEVICES=1 "$PRISM/llama-server" -m "$gg" --host 127.0.0.1 --port 8131 -ngl 99 -c 16384 -np 4 --no-warmup > $W/server.$tag.log 2>&1 &
  SP=$!; i=0
  until curl -sf http://127.0.0.1:8131/health 2>/dev/null | grep -q '"ok"'; do
    i=$((i+1)); kill -0 $SP 2>/dev/null || { echo "FATAL server $tag died" >> $W/probe.log; tail -3 $W/server.$tag.log >> $W/probe.log; exit 1; }
    [ $i -gt 240 ] && { echo "FATAL server $tag not ready" >> $W/probe.log; kill $SP; exit 1; }; sleep 5
  done
  $PY /srv/ml/scripts/bonsai_identity_probe.py --url http://127.0.0.1:8131 --tag $tag --out $W/probe.jsonl >> $W/probe.log 2>&1
  kill $SP; wait $SP 2>/dev/null
done
echo "=== IDENTITY_PROBE_DONE ===" >> $W/probe.log
