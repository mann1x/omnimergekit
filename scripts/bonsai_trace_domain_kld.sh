#!/usr/bin/env bash
# Score ternary arms on HELD-OUT base-model reasoning traces (prompts disjoint from the calibration split),
# next to the wiki.test numbers: is the wiki ordering of the calibration arms a domain effect?
set -uo pipefail
ts(){ date -u '+%H:%M:%SZ'; }
PY=/root/anaconda3/envs/omnimergekit/bin/python
PRISM=/srv/ml/bonsai-assess/ref-b10754/llama-prism-b10754-2459f68
D=/mnt/sdc/opencoti-models/sdc-ml/qwen38-base; GG=$D/Qwen3.8-27B-BF16.gguf
W=/mnt/sdc/opencoti-scratch/v6-ternary/trace_eval; mkdir -p "$W"; TEXT=$W/heldout_traces.txt
if ! grep -q TRACES_DONE "$W/gen.log" 2>/dev/null; then
  LD_LIBRARY_PATH="$PRISM" CUDA_VISIBLE_DEVICES=1 "$PRISM/llama-server" -m "$GG" --host 127.0.0.1 --port 8131 \
      -ngl 99 -c 40960 -np 8 --no-warmup > "$W/server.log" 2>&1 &
  SP=$!
  i=0; until curl -sf http://127.0.0.1:8131/health 2>/dev/null | grep -q '"ok"'; do
    i=$((i+1)); kill -0 $SP 2>/dev/null || { echo "FATAL server died"; exit 1; }
    [ $i -ge 120 ] && { echo "FATAL server not ready"; kill $SP; exit 1; }; sleep 5; done
  "$PY" -u /srv/ml/scripts/bonsai_gen_self_traces.py --corpus /mnt/sdc/ml/omnimerge-v6/calib_train.txt --out "$TEXT" \
      --expect-model Qwen3.8-27B-BF16.gguf --skip 400 --n 64 > "$W/gen.log" 2>&1
  kill $SP; wait $SP 2>/dev/null
  grep -q TRACES_DONE "$W/gen.log" || { echo "FATAL trace generation"; tail -5 "$W/gen.log"; exit 1; }
fi
echo "[$(ts)] $(tail -1 "$W/gen.log")"
i=0; until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -lt 2000 ]; do i=$((i+1)); [ $i -ge 40 ] && { echo "FATAL GPU1 busy"; exit 1; }; sleep 3; done
ppl(){ LD_LIBRARY_PATH="$PRISM" CUDA_VISIBLE_DEVICES=1 "$PRISM/llama-perplexity" "$@" -f "$TEXT" -c 2048 --chunks 16 -b 2048 -ub 512 -ngl 99; }
ppl -m "$GG" --kl-divergence-base "$W/base.kld" > "$W/base.log" 2>&1 || { echo "FATAL base logits"; tail -5 "$W/base.log"; exit 1; }
echo "[$(ts)] base: $(grep -E 'Final estimate' "$W/base.log")"
for A in "prismml:/mnt/sdc/opencoti-models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf" "mixed-corpus:$D/Qwen3.8-27B-pt2-PQ2_0.gguf" "self-traces:$D/Qwen3.8-27B-pt2self-PQ2_0.gguf"; do
  N=${A%%:*}; ppl -m "${A#*:}" --kl-divergence-base "$W/base.kld" --kl-divergence > "$W/kld-$N.log" 2>&1 || tail -4 "$W/kld-$N.log"
  echo "[$(ts)] --- $N ---"; grep -E "^(Mean  *PPL\(Q\) |Mean  *KLD|Same top p)" "$W/kld-$N.log" | sed 's/^/     /'
done
rm -f "$W/base.kld"
echo "[$(ts)] === TRACE_EVAL_DONE ==="
