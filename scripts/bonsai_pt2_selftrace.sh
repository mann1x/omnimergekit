#!/usr/bin/env bash
# AYOT arm of the PT2-style ternarization: calibrate on the base model's own reasoning traces.
# One variable vs the first run: the calibration text. Same n-calib, ctx, method.
set -uo pipefail
ts(){ date -u '+%H:%M:%SZ'; }
PY=/root/anaconda3/envs/omnimergekit/bin/python
PRISM=/srv/ml/bonsai-assess/ref-b10754/llama-prism-b10754-2459f68
GG=/mnt/sdc/opencoti-models/sdc-ml/qwen38-base/Qwen3.8-27B-BF16.gguf
W=/mnt/sdc/opencoti-scratch/v6-ternary/pt2_selftrace; mkdir -p "$W/q"
TR=$W/self_traces.txt
if ! grep -q TRACES_DONE "$W/gen.log" 2>/dev/null; then
  LD_LIBRARY_PATH="$PRISM" CUDA_VISIBLE_DEVICES=1 "$PRISM/llama-server" -m "$GG" --host 127.0.0.1 --port 8131 \
      -ngl 99 -c 40960 -np 8 --no-warmup > "$W/server.log" 2>&1 &
  SP=$!
  i=0; until curl -sf http://127.0.0.1:8131/health 2>/dev/null | grep -q '"ok"'; do
    i=$((i+1)); kill -0 $SP 2>/dev/null || { echo "FATAL server died"; tail -5 "$W/server.log"; exit 1; }
    [ $i -ge 120 ] && { echo "FATAL server not ready in 10 min"; kill $SP; exit 1; }; sleep 5; done
  echo "[$(ts)] base served (pid $SP)"
  "$PY" -u /srv/ml/scripts/bonsai_gen_self_traces.py --corpus /mnt/sdc/ml/omnimerge-v6/calib_train.txt --out "$TR" \
      --expect-model Qwen3.8-27B-BF16.gguf > "$W/gen.log" 2>&1
  kill $SP; wait $SP 2>/dev/null
  grep -q TRACES_DONE "$W/gen.log" || { echo "FATAL trace generation"; tail -6 "$W/gen.log"; exit 1; }
fi
echo "[$(ts)] $(tail -1 "$W/gen.log")  $(stat -c%s "$TR") B"
i=0; until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -lt 2000 ]; do i=$((i+1)); [ $i -ge 40 ] && { echo "FATAL GPU1 not released"; exit 1; }; sleep 3; done
cd /srv/ml/scripts && "$PY" -u bonsai_pt2_full.py --model-dir /mnt/sdc/opencoti-models/hf/Qwen/Qwen3.8-27B \
    --prism-gguf /mnt/sdc/opencoti-models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf --calib-text "$TR" \
    --qdir "$W/q" --device cuda:1 > "$W/run.log" 2>&1
PT2_DIR=$W PT2_TAG=pt2self bash /srv/ml/scripts/bonsai_pt2_export.sh
