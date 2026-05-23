#!/usr/bin/env bash
# stack_canary_4doc_docker.sh — in-container variant of stack_canary_4doc_run.sh.
#
# Designed to run INSIDE the omk-eval-rig container started via
# spool/eval_docker/run.sh. The host wrapper exports VLLM_WHEEL pointing
# at a host-staged wheel (already mounted at /spool/volumes/wheels/).
#
# Phases:
#   1. pip-install $VLLM_WHEEL into the container's `vllm` conda env.
#   2. STACK fingerprint dump (so the H3 result file is forensically
#      complete vs the host stack@1 / stack@2 archives).
#   3. Boot vLLM with the canonical Gemma 4 + reasoning-parser flags.
#   4. Wait for /v1/models, run a tiny smoke, then the 4-doc canary.
#   5. SIGTERM vLLM on exit.
#
# Usage (called from host via run.sh):
#   bash /srv/.../eval_docker/run.sh \
#       --wheel /srv/.../wheels/vllm-0.20.2-cp38-abi3-manylinux_2_35_x86_64.whl \
#       -- bash /omk/scripts/stack_canary_4doc_docker.sh <label>
#
# Exits: 0 ALL_PASS, 2 ANY_FAIL, 3 SETUP_ERROR.

set -uo pipefail

LABEL="${1:?stack label required (e.g. h3_pypi_0.20.2_in_docker)}"

# In-container paths (per spool/eval_docker/run.sh bind mounts)
ROOT="/backup_models"
OMK="/omk"
SPOOL="/spool"

VLLM_PY=/opt/conda/envs/vllm/bin/python
VLLM_PIP=/opt/conda/envs/vllm/bin/pip
OMK_PY=/opt/conda/envs/omnimergekit/bin/python

MODEL_DIR="$ROOT/google/gemma-4-A4B-98e-v5-coder-NVFP4A16"
TOKENIZER="$ROOT/google/gemma-4-26B-A4B-it"
SERVED="98e_v5_coder_nvfp4a16"
PORT=8195

LOGTS=$(date +%Y%m%d_%H%M%S)
OUTDIR="$SPOOL/volumes/eval_results/4doc_${LABEL}_${LOGTS}"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/run.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== H3 in-container canary ($LABEL) ==="
log "outdir=$OUTDIR"
log "model_dir=$MODEL_DIR"

# Phase 1: wheel install (only if a wheel is given)
if [ -n "${VLLM_WHEEL:-}" ]; then
    if [ ! -f "$VLLM_WHEEL" ]; then
        log "ABORT: VLLM_WHEEL=$VLLM_WHEEL not found inside container"
        exit 3
    fi
    log ""
    log "=== Phase 1: pip install $VLLM_WHEEL ==="
    "$VLLM_PIP" install --force-reinstall --no-cache-dir "$VLLM_WHEEL" 2>&1 | tee -a "$LOG" | tail -50
    INSTALL_RC=${PIPESTATUS[0]}
    if [ "$INSTALL_RC" -ne 0 ]; then
        log "ABORT: vllm wheel install failed (rc=$INSTALL_RC)"
        exit 3
    fi
else
    log "(no VLLM_WHEEL set — using whatever vllm is already in the env)"
fi

# Phase 2: STACK fingerprint
log ""
log "=== Phase 2: STACK fingerprint ==="
{
    echo "stack_label=$LABEL"
    echo "timestamp=$LOGTS"
    echo "context=in-container (omk-eval-rig)"
    echo "model_dir=$MODEL_DIR"
    echo "served_name=$SERVED"
    echo "tokenizer=$TOKENIZER"
    echo ""
    echo "--- container OS ---"
    cat /etc/os-release | grep -E "^(NAME|VERSION_ID)=" 2>/dev/null
    ldd --version | head -1
    echo ""
    echo "--- vllm pip ---"
    "$VLLM_PIP" show vllm 2>/dev/null | grep -E "^(Name|Version|Location)"
    echo ""
    echo "--- vllm import sanity + commit ---"
    "$VLLM_PY" -c "import vllm; print('vllm.__version__=', vllm.__version__); print('vllm.__commit__=', getattr(vllm, '__commit__', '(none)'))" 2>&1 | head -5
    echo ""
    echo "--- gemma4_reasoning_parser sha ---"
    PARSER_FILE=$("$VLLM_PY" -c "import vllm.reasoning.gemma4_reasoning_parser as m; print(m.__file__)" 2>/dev/null)
    if [ -n "$PARSER_FILE" ]; then
        echo "parser_file=$PARSER_FILE"
        sha256sum "$PARSER_FILE" 2>/dev/null
    else
        echo "(no gemma4_reasoning_parser found in this build)"
    fi
    echo ""
    echo "--- torch + cuda ---"
    "$VLLM_PY" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.version.cuda)" 2>&1
    echo ""
    echo "--- flashinfer ---"
    "$VLLM_PIP" show flashinfer-python 2>/dev/null | grep -E "^(Name|Version)" || echo "(flashinfer not installed)"
    echo ""
    echo "--- nvidia driver ---"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1
} > "$OUTDIR/STACK.txt" 2>&1
cat "$OUTDIR/STACK.txt" | tee -a "$LOG"

# Synthesize preprocessor_config.json if missing (Gemma 4 + vLLM 0.20.2 quirk).
PROC_CFG="$MODEL_DIR/preprocessor_config.json"
if [ ! -f "$PROC_CFG" ]; then
    log "Synthesizing preprocessor_config.json from processor_config.json"
    "$OMK_PY" - <<PY 2>&1 | tee -a "$LOG"
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

# Phase 3: launch vLLM
log ""
log "=== Phase 3: launching vLLM (background) ==="
SERVER_LOG="$OUTDIR/vllm_server.log"
"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
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
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null | tee -a "$LOG" || true
}
trap cleanup EXIT

log "Waiting for vLLM to be ready (up to 240s)..."
READY=0
for i in $(seq 1 48); do
    if curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        log "vLLM ready after ${i}*5s"
        READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        log "vLLM PID $VLLM_PID died early — see $SERVER_LOG"
        tail -40 "$SERVER_LOG" | tee -a "$LOG"
        exit 3
    fi
    sleep 5
done

if [ "$READY" -ne 1 ]; then
    log "vLLM did not become ready within 240s"
    tail -40 "$SERVER_LOG" | tee -a "$LOG"
    exit 3
fi

log ""
log "=== Phase 4a: smoke (1 trivial req) ==="
curl -fsS "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 5 words.\"}],\"max_tokens\":64,\"temperature\":0}" \
    | tee -a "$LOG" | head -c 600
echo "" | tee -a "$LOG"

log ""
log "=== Phase 4b: 4-doc rumination canary ==="
"$OMK_PY" "$OMK/eval/canary_ifeval_rumination4.py" \
    --base-url "http://localhost:$PORT/v1" \
    --served-name "$SERVED" \
    --out "$OUTDIR/canary_result.json" \
    2>&1 | tee -a "$LOG"
CANARY_RC=${PIPESTATUS[0]}

log ""
log "=== Canary exit code: $CANARY_RC ==="
case $CANARY_RC in
    0) log "RESULT: ALL_PASS - stack $LABEL produces clean answers on all 4 docs." ;;
    2) log "RESULT: ANY_FAIL - stack $LABEL still ruminates on at least one doc." ;;
    3) log "RESULT: SETUP_ERROR - see $OUTDIR/run.log for details." ;;
    *) log "RESULT: unexpected exit $CANARY_RC" ;;
esac

exit $CANARY_RC
