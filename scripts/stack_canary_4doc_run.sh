#!/usr/bin/env bash
# stack_canary_4doc_run.sh — full driver for the 4-doc IFEval rumination canary.
#
# Spins up vLLM with the current installed stack against v5-coder NVFP4A16
# (the canonical canary model per stack.lock.yaml § applies_to), runs the
# 4-prompt canary, then tears the server down cleanly. Use it to iterate
# on stack@N candidates without polluting the cohort results dir.
#
# Usage:
#   bash scripts/stack_canary_4doc_run.sh [stack_label]
#
# Examples:
#   bash scripts/stack_canary_4doc_run.sh stack2_baseline_validate
#   bash scripts/stack_canary_4doc_run.sh stack3_revert_39917
#
# Output: ROOT/canary_results/4doc_<stack_label>_<TS>/canary_result.json + .log
# Exit:   0 ALL_PASS, 2 ANY_FAIL, 3 SETUP_ERROR

set -uo pipefail

ROOT="/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models"
OMK="/shared/dev/omnimergekit"

MODEL_DIR="$ROOT/google/gemma-4-A4B-98e-v5-coder-NVFP4A16"
TOKENIZER="$ROOT/google/gemma-4-26B-A4B-it"
SERVED="98e_v5_coder_nvfp4a16"
PORT=8195

STACK_LABEL="${1:-current}"
LOGTS=$(date +%Y%m%d_%H%M%S)
OUTDIR="$ROOT/canary_results/4doc_${STACK_LABEL}_${LOGTS}"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/run.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Fingerprint the stack so we know what was running. EVAL_PROTOCOL § STACK.txt.
log "=== STACK FINGERPRINT ($STACK_LABEL) ==="
{
    echo "stack_label=$STACK_LABEL"
    echo "timestamp=$LOGTS"
    echo "model_dir=$MODEL_DIR"
    echo "served_name=$SERVED"
    echo "tokenizer=$TOKENIZER"
    echo ""
    echo "--- vllm pip ---"
    /root/anaconda3/envs/vllm/bin/pip show vllm 2>/dev/null | grep -E "^(Name|Version|Location)" || echo "vllm not installed"
    echo ""
    echo "--- vllm wheel commit (if from source) ---"
    /root/anaconda3/envs/vllm/bin/python -c "import vllm; print(getattr(vllm, '__commit__', '(no __commit__)'))" 2>&1 | head -3
    echo ""
    echo "--- gemma4_reasoning_parser file sha (if present) ---"
    PARSER_FILE=$(/root/anaconda3/envs/vllm/bin/python -c "import vllm.reasoning.gemma4_reasoning_parser as m; print(m.__file__)" 2>/dev/null)
    if [ -n "$PARSER_FILE" ]; then
        echo "parser_file=$PARSER_FILE"
        sha256sum "$PARSER_FILE" 2>/dev/null
    fi
    echo ""
    echo "--- torch + cuda ---"
    /root/anaconda3/envs/vllm/bin/python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.version.cuda)" 2>&1
    echo ""
    echo "--- nvidia driver ---"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1
    echo ""
    echo "--- stack.lock.yaml version line ---"
    grep -E "^(name|version|created):" "$OMK/eval/stack.lock.yaml" 2>/dev/null
} > "$OUTDIR/STACK.txt" 2>&1
cat "$OUTDIR/STACK.txt" | tee -a "$LOG"

# Pre-flight: GPU should be idle
log ""
log "=== GPU state ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$LOG"
GPU_USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${GPU_USED_MB:-0}" -gt 2000 ]; then
    log "ABORT: GPU has ${GPU_USED_MB} MiB in use — another process running. Free GPU then retry."
    exit 3
fi

# Synthesize preprocessor_config.json if missing (Gemma 4 + vLLM 0.20.2 quirk).
PROC_CFG="$MODEL_DIR/preprocessor_config.json"
if [ ! -f "$PROC_CFG" ]; then
    log "Synthesizing preprocessor_config.json from processor_config.json"
    /root/anaconda3/envs/omnimergekit/bin/python - <<PY 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
src = Path("$MODEL_DIR/processor_config.json")
dst = Path("$PROC_CFG")
if src.exists():
    cfg = json.loads(src.read_text())
    fe = cfg.get("feature_extractor", {})
    if fe:
        dst.write_text(json.dumps(fe, indent=2))
        print(f"wrote {dst} from feature_extractor block")
    else:
        print("processor_config.json has no feature_extractor — skipping")
else:
    print(f"no processor_config.json at {src} — skipping")
PY
fi

log ""
log "=== Launching vLLM server (background) ==="
SERVER_LOG="$OUTDIR/vllm_server.log"
/root/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$SERVED" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --dtype bfloat16 \
    --trust-remote-code \
    --reasoning-parser gemma4 \
    --tokenizer "$TOKENIZER" \
    > "$SERVER_LOG" 2>&1 &
VLLM_PID=$!
log "vLLM PID=$VLLM_PID (log: $SERVER_LOG)"
disown $VLLM_PID

cleanup() {
    log "=== Cleanup: SIGTERM vLLM PID $VLLM_PID ==="
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill -TERM "$VLLM_PID" 2>/dev/null
        for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$VLLM_PID" 2>/dev/null || true
    fi
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$LOG"
}
trap cleanup EXIT

log "Waiting for vLLM to be ready (up to 240s)..."
READY=0
for i in $(seq 1 48); do
    if curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        log "vLLM ready after ${i}×5s"
        READY=1
        break
    fi
    # Bail early if vLLM died
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        log "vLLM PID $VLLM_PID died early — see $SERVER_LOG"
        tail -30 "$SERVER_LOG" | tee -a "$LOG"
        exit 3
    fi
    sleep 5
done

if [ "$READY" -ne 1 ]; then
    log "vLLM did not become ready within 240s"
    tail -30 "$SERVER_LOG" | tee -a "$LOG"
    exit 3
fi

# Tiny smoke before the canary (ifeval_100 doc 0 — "Why is everyone…")
log ""
log "=== Smoke (1 trivial req) ==="
curl -fsS "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 5 words.\"}],\"max_tokens\":64,\"temperature\":0}" \
    | tee -a "$LOG" | head -c 600
echo "" | tee -a "$LOG"

log ""
log "=== Running 4-doc canary ==="
/root/anaconda3/envs/omnimergekit/bin/python "$OMK/eval/canary_ifeval_rumination4.py" \
    --base-url "http://localhost:$PORT/v1" \
    --served-name "$SERVED" \
    --out "$OUTDIR/canary_result.json" \
    2>&1 | tee -a "$LOG"
CANARY_RC=${PIPESTATUS[0]}

log ""
log "=== Canary exit code: $CANARY_RC ==="
case $CANARY_RC in
    0) log "RESULT: ALL_PASS — stack $STACK_LABEL produces clean answers on all 4 docs." ;;
    2) log "RESULT: ANY_FAIL — stack $STACK_LABEL still ruminates on at least one doc." ;;
    3) log "RESULT: SETUP_ERROR — see $OUTDIR/run.log for details." ;;
    *) log "RESULT: unexpected exit $CANARY_RC" ;;
esac

# Cleanup runs via trap
exit $CANARY_RC
