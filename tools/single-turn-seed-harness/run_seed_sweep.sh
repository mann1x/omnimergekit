#!/usr/bin/env bash
# run_seed_sweep.sh -- OLD-vs-NEW Gemma-4 chat-template single-turn seed sweep.
#
# Serves the 12B F16 GGUF twice on GPU0 (OLD google_main, then NEW
# google_20260709), runs the single-turn HumanEval+/MultiPL-E seed sweep against
# each, kills each server by its explicit PID, and prints the comparison table.
#
# HARD DISCIPLINE (see the omnimergekit CLAUDE.md / task brief):
#   * GPU0 ONLY -- every llama-server launch pins CUDA_VISIBLE_DEVICES=0.
#     GPU1 runs a training job; never touch it.
#   * Never touch the opencoti supervisor on :8240 or sshd. This uses PORT
#     (default 8097 -- 8090 is often already taken on bs2).
#   * Kill llama-server by the captured $! PID ONLY. Never `pkill -f`.
#   * No large files in /tmp -- logs + results go under this tool's results/.
#
# Usage:
#   bash run_seed_sweep.sh                # full sweep (all HE+ + MPE, 12 seeds)
#   SMOKE=1 bash run_seed_sweep.sh        # 2-problem health smoke, 1 seed
#   TASKS='he:40,cpp:20,js:20' SEEDS=12 bash run_seed_sweep.sh   # custom scale
set -uo pipefail

# --- config (override via env) ---------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-/mnt/sdc/ml/llama.cpp-latest/build/bin/llama-server}"
GGUF="${GGUF:-/mnt/sdc/ml/google/gemma-4-12B-it-F16.gguf}"
TPL_DIR="${TPL_DIR:-/srv/ml/repos/omnimergekit/tools/agentic-loop-harness/templates}"
TPL_OLD="${TPL_OLD:-$TPL_DIR/google_main.jinja}"
TPL_NEW="${TPL_NEW:-$TPL_DIR/google_20260709.jinja}"
PYTHON="${PYTHON:-/root/anaconda3/envs/omnimergekit/bin/python}"
PORT="${PORT:-8097}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-65536}"
PARALLEL="${PARALLEL:-4}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
REASON_BUDGET="${REASON_BUDGET:-16384}"
SEEDS="${SEEDS:-12}"
SEED0="${SEED0:-2000}"
TASKS="${TASKS:-all}"
CONCURRENCY="${CONCURRENCY:-$PARALLEL}"
OUT="${OUT:-$HERE/results}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  TASKS="${SMOKE_TASKS:-he:2}"; SEEDS=1; CONCURRENCY=1
  OUT="$HERE/results/smoke"
  echo ">>> SMOKE mode: tasks=$TASKS seeds=$SEEDS"
fi

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
[[ -f /root/.cache/huggingface/token ]] && export HF_TOKEN="$(cat /root/.cache/huggingface/token)"

mkdir -p "$OUT"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

for f in "$LLAMA_SERVER" "$GGUF" "$TPL_OLD" "$TPL_NEW" "$PYTHON"; do
  [[ -e "$f" ]] || { echo "FATAL: missing $f" >&2; exit 1; }
done

# refuse to collide with the production supervisor or an in-use port
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "FATAL: port $PORT is already in use -- pick another PORT (NOT 8240/8090)." >&2
  exit 1
fi

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ">>> killing llama-server PID $SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

serve() {  # serve <template-file> <log>
  local tpl="$1" log="$2"
  echo ">>> serving $(basename "$tpl") on GPU0 :$PORT  (log: $log)"
  CUDA_VISIBLE_DEVICES=0 "$LLAMA_SERVER" \
    -m "$GGUF" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c "$CTX" -np "$PARALLEL" \
    --jinja --chat-template-file "$tpl" \
    --reasoning-format deepseek --reasoning-budget "$REASON_BUDGET" \
    --host "$HOST" --port "$PORT" >"$log" 2>&1 &
  SERVER_PID=$!
  disown "$SERVER_PID" 2>/dev/null || true
  echo ">>> llama-server PID=$SERVER_PID  waiting for /health ..."
  for _ in $(seq 1 180); do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "FATAL: server died early; tail:"; tail -30 "$log"; exit 1; }
    if curl -sf "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
      echo ">>> health OK (PID $SERVER_PID)"; return 0
    fi
    sleep 2
  done
  echo "FATAL: server never became healthy; tail:"; tail -40 "$log"; exit 1
}

sweep() {  # sweep <name>
  local name="$1"
  echo ">>> sweep --name $name --tasks $TASKS --seeds $SEEDS"
  "$PYTHON" -m single_turn_seed_harness.cli \
    --server "http://$HOST:$PORT" --name "$name" \
    --tasks "$TASKS" --seeds "$SEEDS" --seed0 "$SEED0" \
    --max-tokens "$MAX_TOKENS" --concurrency "$CONCURRENCY" --out "$OUT"
}

# --- OLD template -----------------------------------------------------------
serve "$TPL_OLD" "$OUT/llama_old-gmain.log"
sweep "old-gmain"
cleanup

# --- NEW template -----------------------------------------------------------
serve "$TPL_NEW" "$OUT/llama_new-g0709.log"
sweep "new-g0709"
cleanup

# --- comparison table -------------------------------------------------------
echo
"$PYTHON" -m single_turn_seed_harness.tabulate \
  "$OUT/summary_old-gmain.json" "$OUT/summary_new-g0709.json"
echo
echo ">>> results in $OUT"
