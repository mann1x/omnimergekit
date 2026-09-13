#!/usr/bin/env bash
# SEQUENTIAL base-cohort wall-time cell.
#
# WHY. The published base cohort ran both arms CONCURRENTLY on separate GPUs of
# one host, so its wall-time comparison is contention-confounded. This re-runs
# the SAME cohort one arm at a time, on the SAME GPU, with nothing else on the
# box, to produce a citable latency number -- and to calibrate the contention
# factor per arm. If both arms inflate by the same factor, the paired-contended
# methodology is validated and the composite wall figure can stand on its
# frozen 120-task balanced subset without a 4.5h re-run.
#
# BASIS: byte-identical to the original base cohort (serve_arm.sh + run_bfcl.sh)
#   serve   -c 131072 -np 4 -ngl 99 --no-warmup -t 16 --jinja
#           --reasoning-format deepseek --reasoning-budget 8192
#   inspect multi_turn_base --limit 200 --max-connections 4 --max-tokens 8192
#           --temperature 0.0
# Changing ANY of these voids the comparison against the contended cohort.
set -uo pipefail
H=/mnt/sdc/harness; L=$H/logs; TP=/mnt/sdc/v7rework/template_probe
LC=/srv/ml/tools/llama.cpp/build-pegfix/bin
GGUF=/srv/ml/models/gguf/google-a4b-128e/google_gemma-4-26B-A4B-it-Q4_K_M-eos106.gguf
GPU=0              # SAME GPU for both arms -- a GPU swap would confound latency
PORT=8302
VENV=$H/venv
LOG=$L/seq_base2.log
exec >> "$LOG" 2>&1
echo "=================================================================="
echo ">>> SEQ BASE CELL PASS-2 (REVERSED ORDER B,A) start $(date -u +%FT%TZ)"

# ---- optional chain wait -----------------------------------------------
# A TIMEOUT HERE MUST ABORT. The 2026-09-12 incident: harbor_smoke.sh waited 6h
# for a marker, fell through when it never came, announced \"composite done\" and
# killed the still-running arm. A wait loop that proceeds on timeout is a gate
# that lies. Set WAIT_FOR=\"marker:file ...\" to arm it; empty = run now.
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
  if [ "$ok" -ne 1 ]; then
    echo "SEQ_BASE2_ABORT chain wait TIMED OUT on: $WAIT_FOR $(date -u +%FT%TZ)"
    echo "                 refusing to proceed on an unmet precondition."
    exit 2
  fi
  echo ">>> chain markers seen $(date -u +%FT%TZ)"
else
  echo ">>> WAIT_FOR empty; running immediately"
fi

# ---- GATE: nothing else may be running, and VRAM must actually be free ----
# Port-close != VRAM release. On 2026-09-12 the smoke server started as soon as
# the ports closed and died with cudaMalloc OOM (35.5G free of 97G) because the
# previous server had not released its buffers yet. Gate on free VRAM.
NEED_MIB=${NEED_MIB:-45000}
idle_ok=0
for i in $(seq 1 120); do
  NS=$(ps -eo args | grep -c "[l]lama-server")
  NI=$(ps -eo args | grep -c "[i]nspect eval")
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -dc 0-9)
  : "${FREE:=0}"
  if [ "$NS" -eq 0 ] && [ "$NI" -eq 0 ] && [ "$FREE" -ge "$NEED_MIB" ]; then idle_ok=1; break; fi
  echo "    waiting for idle: llama-server=$NS inspect=$NI gpu${GPU}_free=${FREE}MiB (need ${NEED_MIB})"
  sleep 30
done
if [ "$idle_ok" -ne 1 ]; then
  echo "SEQ_BASE2_ABORT box never went idle $(date -u +%FT%TZ)"; exit 1
fi
echo ">>> isolation gate PASSED (gpu${GPU} free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")MiB)"

run_one () {   # run_one <arm> <template>
  local ARM="$1" TPL="$2"
  local TAG="seq_${ARM}"
  echo ">>> --- $TAG start $(date -u +%FT%TZ) ---"
  nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw,utilization.gpu --format=csv,noheader | sed "s/^/    gpu-at-start /"
  CUDA_VISIBLE_DEVICES="$GPU" LD_LIBRARY_PATH="$LC:${LD_LIBRARY_PATH:-}" \
  nohup "$LC/llama-server" -m "$GGUF" --host 127.0.0.1 --port "$PORT" \
    -c 131072 -np 4 -ngl 99 --no-warmup -t 16 \
    --jinja --chat-template-file "$TPL" \
    --reasoning-format deepseek --reasoning-budget 8192 \
    > "$L/server_${TAG}.log" 2>&1 &
  local SP=$!
  disown
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && break
    sleep 5
  done
  if [ "$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null)" != "200" ]; then
    echo "SEQ_BASE_FAIL $TAG server never healthy $(date -u +%FT%TZ)"; kill "$SP" 2>/dev/null; return 1
  fi
  local D="$L/bfcl_${TAG}"
  mkdir -p "$D"
  bash "$H/write_stack.sh" "$TAG" "$D" "$TPL" "$PORT" >/dev/null 2>&1 || echo "    WARN no STACK.txt"
  local T0=$(date +%s)
  HF_HOME=$H/hf LLAMACPP_BASE_URL="http://127.0.0.1:$PORT/v1" LLAMACPP_API_KEY=none \
  INSPECT_LOG_DIR="$D" \
  "$VENV/bin/inspect" eval inspect_evals/bfcl \
    -T categories=multi_turn_base \
    --model "openai-api/llamacpp/gemma-4-26B-A4B-it-${TAG}" \
    --limit 200 --max-connections 4 --max-tokens 8192 --temperature 0.0 \
    --log-dir "$D" --no-fail-on-error 2>&1 | tail -25
  local T1=$(date +%s)
  echo ">>> $TAG ELAPSED_WALL_S=$(( T1 - T0 ))"; nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw --format=csv,noheader | sed "s/^/    gpu-at-end /"
  kill "$SP" 2>/dev/null
  for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 3; done
  echo "SEQ_BASE_${ARM}_DONE $(date -u +%FT%TZ)"
}

run_one armB2 "$TP/v7coder_chat_template.jinja"
run_one armA2 "$TP/google_live_chat_template.jinja"
echo "SEQ_BASE2_CELL_DONE $(date -u +%FT%TZ)"
