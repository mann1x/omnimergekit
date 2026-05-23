#!/usr/bin/env bash
# stack_canary_4doc_llama.sh — run the 4-doc IFEval rumination canary against
# llama-server (instead of vLLM). Used to isolate whether the rumination
# regression seen with vLLM is a vLLM-stack artifact or a model artifact.
#
# Usage:
#   bash stack_canary_4doc_llama.sh <label> <gguf-path>
#
# Boots llama-server on :8195 (same port the canary script defaults to),
# waits for /v1/models, runs the canary, then SIGTERM'd llama-server on exit.

set -uo pipefail
LABEL="${1:?label required}"
GGUF="${2:?gguf path required}"

[ -f "$GGUF" ] || { echo "ABORT: gguf not found: $GGUF" >&2; exit 3; }

PORT=8195
SERVED=v5coder_q6k_llama
LLAMA=/opt/llama.cpp/build/bin/llama-server

CANARY=/shared/dev/omnimergekit/eval/canary_ifeval_rumination4.py
PY=/root/anaconda3/envs/omnimergekit/bin/python

OUT=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/canary_results/4doc_${LABEL}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

echo "[$(date +%H:%M:%S)] === llama-server boot ==="
echo "gguf:    $GGUF"
echo "served:  $SERVED  port=$PORT"
echo "out:     $OUT"

# Boot llama-server in background — Gemma 4 mandatory flags from CLAUDE.md
"$LLAMA" -m "$GGUF" --port "$PORT" -c 32768 -t 12 -ngl 99 --no-warmup \
    --reasoning-format deepseek --reasoning-budget 12288 \
    --alias "$SERVED" --jinja \
    > "$OUT/llama_server.log" 2>&1 &
LLAMA_PID=$!
echo "[$(date +%H:%M:%S)] llama-server pid=$LLAMA_PID"

cleanup() {
    if [ -n "${LLAMA_PID:-}" ] && kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] === Cleanup: SIGTERM llama-server PID $LLAMA_PID ==="
        kill -TERM "$LLAMA_PID" 2>/dev/null
        for i in 1 2 3 4 5; do
            kill -0 "$LLAMA_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$LLAMA_PID" 2>/dev/null || true
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1
    fi
}
trap cleanup EXIT

# Wait for /v1/models
echo "[$(date +%H:%M:%S)] waiting for /v1/models ..."
for i in $(seq 1 60); do
    if curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] /v1/models reachable after ${i}*5s"
        break
    fi
    sleep 5
done
curl -fsS "http://localhost:$PORT/v1/models" >/dev/null || { echo "ABORT: llama-server never came up"; exit 4; }

# Sanity ping — short greedy completion to confirm chat-template + reasoning
curl -sS "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"temperature\":0,\"max_tokens\":50}" \
    | tee "$OUT/sanity.json" | head -c 500
echo ""

echo "[$(date +%H:%M:%S)] === Running 4-doc canary ==="
"$PY" "$CANARY" \
    --base-url "http://localhost:$PORT/v1" \
    --served-name "$SERVED" \
    --out "$OUT/canary_result.json" \
    --max-tokens 16384 \
    --thinking-budget 12288 \
    --timeout 300 \
    2>&1 | tee "$OUT/canary.log"
RC=${PIPESTATUS[0]}

echo "[$(date +%H:%M:%S)] canary rc=$RC"
case "$RC" in
    0) VERDICT="ALL_PASS" ;;
    2) VERDICT="ANY_FAIL" ;;
    3) VERDICT="SETUP_ERROR" ;;
    *) VERDICT="UNEXPECTED_RC_$RC" ;;
esac
echo "[$(date +%H:%M:%S)] FINAL VERDICT for $LABEL: $VERDICT — see $OUT/canary_result.json"
exit $RC
