#!/usr/bin/env bash
# Harbor SMOKE: 2 compilebench tasks, one arm, to validate
#   (a) container -> llama-server on the docker bridge
#   (b) the harbor jobs-dir layout that score_harbor() must parse
#
# REWRITTEN 2026-09-13 after the previous version destroyed a running experiment.
# Three defects, all fixed here:
#   1. Its wait loop FELL THROUGH on timeout: after 6h it printed "composite
#      done" and killed the 8201/8202 servers while armA was still running.
#      A gate that proceeds on an unmet precondition is a gate that lies.
#      -> WAIT_FOR now ABORTS on timeout, and this script NEVER kills anything.
#   2. It started the server as soon as the ports closed; a closed port is not a
#      freed buffer, and llama-server died with cudaMalloc OOM (35.5G free of
#      97G).  -> gate on FREE VRAM.
#   3. `proactive_summarize_threshold` is not a terminus-2 option; the real name
#      is `proactive_summarization_threshold`. harbor rejected the whole run.
set -uo pipefail
L=/mnt/sdc/harness/logs
TP=/mnt/sdc/v7rework/template_probe
GGUF=/srv/ml/models/gguf/google-a4b-128e/google_gemma-4-26B-A4B-it-Q4_K_M-eos106.gguf
LC=/srv/ml/tools/llama.cpp/build-pegfix/bin
BRIDGE=172.17.0.1
PORT=8401
GPU=0
OUT=/mnt/sdc/agentbench/smoke
NEED_MIB=${NEED_MIB:-45000}

WAIT_FOR="${WAIT_FOR:-}"
if [ -n "$WAIT_FOR" ]; then
  ok=0
  for i in $(seq 1 720); do
    all=1
    for spec in $WAIT_FOR; do
      k=${spec%%:*}; f=${spec#*:}
      n=$(grep -ac "$k" "$f" 2>/dev/null); n=${n:-0}
      [ "$n" -ge 1 ] || all=0
    done
    [ "$all" -eq 1 ] && { ok=1; break; }
    sleep 30
  done
  [ "$ok" -eq 1 ] || { echo "SMOKE_ABORT chain wait TIMED OUT on: $WAIT_FOR $(date -u +%FT%TZ)"; exit 2; }
  echo ">>> chain markers seen $(date -u +%FT%TZ)"
fi

# NEVER kill someone else s work. Wait for the box, or abort.
vram_ok=0
for i in $(seq 1 240); do
  NS=$(ps -eo args | grep -c "[l]lama-server")
  NI=$(ps -eo args | grep -c "[i]nspect eval")
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -dc "0-9"); : "${FREE:=0}"
  if [ "$NS" -eq 0 ] && [ "$NI" -eq 0 ] && [ "$FREE" -ge "$NEED_MIB" ]; then vram_ok=1; break; fi
  echo "    waiting: llama-server=$NS inspect=$NI gpu${GPU}_free=${FREE}MiB"
  sleep 30
done
[ "$vram_ok" -eq 1 ] || { echo "SMOKE_ABORT box never idle $(date -u +%FT%TZ)"; exit 1; }

mkdir -p "$OUT"
echo ">>> serving on docker bridge $BRIDGE:$PORT (NOT 0.0.0.0 - no auth, public iface)"
CUDA_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH="$LC:${LD_LIBRARY_PATH:-}" \
nohup "$LC/llama-server" -m "$GGUF" --host "$BRIDGE" --port "$PORT" \
  -c 262144 -np 2 -ngl 99 --no-warmup -t 16 \
  --jinja --chat-template-file "$TP/google_live_chat_template.jinja" \
  --reasoning-format auto --reasoning-budget -1 \
  > "$OUT/server_smoke.log" 2>&1 &
SP=$!
disown
for i in $(seq 1 240); do
  [ "$(curl -s -o /dev/null -w "%{http_code}" "http://$BRIDGE:$PORT/health" 2>/dev/null)" = "200" ] && break
  sleep 5
done
HC=$(curl -s -o /dev/null -w "%{http_code}" "http://$BRIDGE:$PORT/health" 2>/dev/null)
if [ "$HC" != "200" ]; then
  echo "SMOKE_FAIL server never healthy (http=$HC). tail:"; tail -12 "$OUT/server_smoke.log"
  kill "$SP" 2>/dev/null; exit 1
fi
echo ">>> health OK (http 200)"

echo ">>> reachability FROM A CONTAINER (the thing that actually matters)"
docker run --rm curlimages/curl:latest \
  -s -m 20 -o /dev/null -w "container->host HTTP %{http_code}\n" \
  "http://$BRIDGE:$PORT/health" 2>&1 | tail -2

echo ">>> harbor smoke: 2 compilebench tasks, terminus-2, greedy, summarize OFF"
cd /mnt/sdc/agentbench
OPENAI_API_KEY=none OPENAI_BASE_URL="http://$BRIDGE:$PORT/v1" \
timeout 3000 /mnt/sdc/agentbench/venv-harbor/bin/harbor run \
  --path /mnt/sdc/agentbench/datasets/compilebench \
  --agent terminus-2 --model "openai/gemma4-128e" \
  --ak "api_base=http://$BRIDGE:$PORT/v1" --ak "temperature=0.0" \
  --ak "enable_summarize=false" --ak "proactive_summarization_threshold=0" \
  --allow-agent-host "$BRIDGE" \
  --jobs-dir "$OUT/jobs" --job-name smoke -n 2 --n-tasks 2 -y 2>&1 | tail -25

echo ">>> JOBS DIR LAYOUT (what score_harbor must parse)"
find "$OUT/jobs" -maxdepth 4 2>/dev/null | head -40
echo ">>> json files"; find "$OUT/jobs" -name "*.json" 2>/dev/null | head -15
kill "$SP" 2>/dev/null
echo "HARBOR_SMOKE_DONE $(date -u +%FT%TZ)"
