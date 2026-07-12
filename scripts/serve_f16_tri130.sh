#!/usr/bin/env bash
# serve dern11-F16 on GPU0:8194, replay HumanEval/130 greedy, PID-kill the server.
set -uo pipefail
LS=/mnt/sdc/ml/llama.cpp-latest/build/bin/llama-server
PY=/root/anaconda3/envs/omnimergekit/bin/python
GGUF=/mnt/sdc/ml/sft_heal/v7coder-dern11-F16.gguf
GPU=0; PORT=8194
ts(){ date "+%T %Z"; }
[ -e "$GGUF" ] || { echo "FATAL missing $GGUF"; exit 2; }
mkdir -p /srv/ml/agentic_loop/logs
echo "[$(ts)] launch F16 server GPU$GPU:$PORT (~40GB weights)"
CUDA_VISIBLE_DEVICES=$GPU nohup "$LS" -m "$GGUF" --host 127.0.0.1 --port $PORT \
  --alias rp --jinja -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 32768 --no-warmup \
  --reasoning-format deepseek --reasoning-budget 12288 \
  > /srv/ml/agentic_loop/logs/f16_srv_${PORT}.log 2>&1 &
SRV=$!
echo "[$(ts)] server PID=$SRV"
trap "kill $SRV 2>/dev/null" EXIT
OK=0
for i in $(seq 1 300); do
  curl -fsS -m 10 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0}" \
    >/dev/null 2>&1 && { echo "[$(ts)] :$PORT SERVING ~$((i*2))s"; OK=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "[$(ts)] SERVER DIED during boot"; tail -30 /srv/ml/agentic_loop/logs/f16_srv_${PORT}.log; exit 3; }
  sleep 2
done
[ $OK = 1 ] || { echo "[$(ts)] server never came up"; exit 3; }
echo "[$(ts)] === tri130_replay against F16 ==="
"$PY" /srv/ml/scripts/tri130_replay.py "http://127.0.0.1:${PORT}"
RC=$?
echo "[$(ts)] replay rc=$RC"
kill $SRV 2>/dev/null; trap - EXIT; sleep 2
echo "[$(ts)] === F16 TRI-130 DONE ==="
