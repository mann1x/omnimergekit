#!/usr/bin/env bash
# Pod-side LCB-medium 55q eval via vLLM 4-bit (bitsandbytes on-load) for:
#   - gemma-4-31b-he1-it    (bf16 at /workspace/replay_prune/pruned.broken)
#   - gemma-4-31b base      (downloaded from google/gemma-3-27b-it if absent)
#
# The script follows the omnimergekit BF16 v3 eval methodology (chat-completions
# endpoint, lm-eval-style harness for code tasks), substituting the
# eval/lcb/lcb_llama_server.py runner which is already OpenAI-compatible.
#
# Trigger condition (handled in wrapper):
#   - the he1quant tmux pipeline has finished imatrix.dat (so GPU is free
#     for vLLM serving while the CPU-bound quants finish in parallel)
#
# Args:
#   $1 — base-model HF id (default: google/gemma-3-27b-it)
#   $2 — vllm port (default 8090)
#
# Outputs:
#   /workspace/eval_lcb55_vllm4bit/{he1,base}/...
#   /workspace/logs/lcb55_vllm_4bit_*.log

set -euo pipefail

WS=/workspace
LOGS=$WS/logs
RESULTS=$WS/eval_lcb55_vllm4bit
mkdir -p "$LOGS" "$RESULTS"

BASE_ID="${1:-google/gemma-4-31B-it}"
PORT="${2:-8090}"
HE1_BF16=$WS/replay_prune/pruned.broken
BASE_BF16=$WS/google_gemma-4-31B-it
HE1_HF_REPO=ManniX-ITA/gemma-4-31b-he1-it

ts()  { date +%Y%m%d_%H%M%S; }
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOGS/lcb55_vllm_4bit_orchestrator.log"; }

# ---------- 1. preflight: vLLM env + lcb runner ----------
PY=/opt/conda/bin/python3
log "preflight: python=$PY"
if ! $PY -c "import vllm" 2>/dev/null; then
    log "  vllm missing — installing (vllm + bitsandbytes) ..."
    $PY -m pip install --quiet vllm bitsandbytes 2>&1 | tail -5 \
        | tee -a "$LOGS/lcb55_vllm_4bit_orchestrator.log"
fi
$PY -c "import vllm, bitsandbytes; print('vllm', vllm.__version__, 'bnb', bitsandbytes.__version__)" \
    2>&1 | tee -a "$LOGS/lcb55_vllm_4bit_orchestrator.log"

# omnimergekit + lcb runner
OMK=$WS/omnimergekit
if [[ ! -d "$OMK/eval/lcb" ]]; then
    log "cloning omnimergekit (need lcb_llama_server.py + eval helpers)"
    git clone --depth 1 https://github.com/mann1x/omnimergekit "$OMK" 2>&1 \
        | tee -a "$LOGS/lcb55_vllm_4bit_orchestrator.log"
fi
LCB=$OMK/eval/lcb/lcb_llama_server.py
[[ -f "$LCB" ]] || { log "FATAL: $LCB missing"; exit 2; }

# ---------- 2. download he1-it bf16 weights if absent ----------
# (base 31B bf16 is downloaded LATER, AFTER he1-it eval finishes and the
# he1-it safetensors are nuked — disk on the pod cannot hold both 62 GB
# bf16 dirs + the parallel quant pipeline's F16 GGUF + in-flight quants.)
if [[ ! -f "$HE1_BF16/model.safetensors.index.json" && ! -f "$HE1_BF16/model-00001-of-00002.safetensors" ]]; then
    log "downloading $HE1_HF_REPO bf16 weights to $HE1_BF16 (~62 GB) ..."
    mkdir -p "$HE1_BF16"
    hf download "$HE1_HF_REPO" --local-dir "$HE1_BF16" --exclude "*.gguf" --exclude "imatrix*" 2>&1 \
        | tail -30 | tee -a "$LOGS/lcb55_vllm_4bit_orchestrator.log"
else
    log "he1-it bf16 already local at $HE1_BF16"
fi

# ---------- 3. helper: serve + eval one variant ----------
run_one () {
    local NAME="$1"      # short tag (he1-it / 31b-base)
    local MODEL_DIR="$2" # local bf16 dir
    local SDIR="$RESULTS/$NAME"
    mkdir -p "$SDIR"

    local SLOG="$LOGS/vllm_${NAME}_$(ts).log"
    log "=== launching vLLM 4-bit (bnb) for $NAME from $MODEL_DIR ==="

    # 31B nf4 ≈ 16-17 GB; KV ≈ 2-3 GB at 16384; --enforce-eager because head_dim=512
    # isn't supported by flash-attn in vLLM 0.19.0 for Gemma 4.
    $PY -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_DIR" \
        --served-model-name "$NAME" \
        --quantization bitsandbytes \
        --load-format bitsandbytes \
        --port $PORT \
        --gpu-memory-utilization 0.92 \
        --max-model-len 16384 \
        --enforce-eager \
        --dtype bfloat16 \
        > "$SLOG" 2>&1 &
    local SPID=$!
    disown

    log "  vllm pid=$SPID slog=$SLOG; waiting up to 12 min for ready ..."
    local ready=0
    for i in $(seq 1 360); do
        if curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
            log "  vLLM ready after ${i}×2s"
            ready=1; break
        fi
        if ! kill -0 "$SPID" 2>/dev/null; then
            log "  ABORT: vllm pid=$SPID died during startup"
            tail -40 "$SLOG" | sed 's/^/    /' | tee -a "$LOGS/lcb55_vllm_4bit_orchestrator.log"
            return 2
        fi
        sleep 2
    done
    if [[ $ready -ne 1 ]]; then
        log "  ABORT: vllm never responded; tail $SLOG:"
        tail -40 "$SLOG" | sed 's/^/    /' | tee -a "$LOGS/lcb55_vllm_4bit_orchestrator.log"
        kill -KILL $SPID 2>/dev/null || true
        return 3
    fi

    # one-shot smoke: confirm the model actually speaks before burning an hour on LCB
    log "  smoke test: prime check via /v1/chat/completions ..."
    curl -sS "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Return ONLY the Python function: def is_prime(n):\"}],\"max_tokens\":120,\"temperature\":0}" \
        > "$SDIR/smoke.json" 2>&1
    head -c 600 "$SDIR/smoke.json"; echo

    # LCB-medium 55q via OpenAI-compatible client (max-tokens 16384, post-2024-10)
    log "  running LCB-medium 55q on $NAME via vLLM bnb-4bit ..."
    local T0=$(date +%s)
    $PY "$LCB" \
        --name "$NAME" --base-url "http://localhost:$PORT" \
        --limit 999 --max-tokens 16384 \
        --output "$SDIR/${NAME}_lcb_med_55q_vllm4bit.json" \
        > "$LOGS/lcb55_vllm_4bit_${NAME}_runner_$(ts).log" 2>&1 || true
    log "  LCB wall=$(($(date +%s)-T0))s"

    # cleanup
    kill -TERM $SPID 2>/dev/null || true
    sleep 5; kill -KILL $SPID 2>/dev/null || true
    pkill -KILL -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null || true
    sleep 8

    # quick score
    python3 - <<PY | tee -a "$LOGS/lcb55_vllm_4bit_orchestrator.log"
import json
try:
    r = json.load(open("$SDIR/${NAME}_lcb_med_55q_vllm4bit.json"))
    print(f"  RESULT [$NAME] vllm-bnb4bit LCB-medium pass@1 = {r['pass_at_1']*100:.2f}%  ({r['n_pass']}/{r['n']})")
except Exception as e:
    print(f"  RESULT [$NAME] ERROR: {e}")
PY
}

# ---------- 4. run both variants ----------
run_one "31b-he1-it" "$HE1_BF16"

# Disk hygiene: nuke he1-it bf16 safetensors before downloading the base bf16
# (each is ~62 GB; pod /workspace cannot hold both + F16 GGUF + a quant in flight).
log "freeing he1-it bf16 safetensors before base bf16 download ..."
rm -f "$HE1_BF16"/*.safetensors
df -h /workspace | tail -1 | tee -a "$LOGS/lcb55_vllm_4bit_orchestrator.log"

# Now download base if absent and run the second eval.
if [[ ! -f "$BASE_BF16/model.safetensors.index.json" || -z "$(ls "$BASE_BF16"/*.safetensors 2>/dev/null | head -1)" ]]; then
    log "downloading $BASE_ID bf16 weights to $BASE_BF16 (~62 GB) ..."
    mkdir -p "$BASE_BF16"
    hf download "$BASE_ID" --local-dir "$BASE_BF16" --exclude "*.gguf" 2>&1 \
        | tail -30 | tee -a "$LOGS/lcb55_vllm_4bit_orchestrator.log"
fi

run_one "31b-base"   "$BASE_BF16"

# Disk hygiene: drop base bf16 too — both evals done, weights still on HF.
log "cleaning up base bf16 safetensors post-eval ..."
rm -f "$BASE_BF16"/*.safetensors

# ---------- 5. side-by-side ----------
log "=== FINAL ==="
python3 - <<'PY' | tee -a "$LOGS/lcb55_vllm_4bit_orchestrator.log"
import json, pathlib
root = pathlib.Path("/workspace/eval_lcb55_vllm4bit")
print(f"{'variant':<14}{'pass@1':<12}{'n_pass/n':<10}{'serving'}")
for tag in ("31b-he1-it", "31b-base"):
    p = root / tag / f"{tag}_lcb_med_55q_vllm4bit.json"
    if p.exists():
        r = json.load(open(p))
        print(f"{tag:<14}{r['pass_at_1']*100:>6.2f}%     {r['n_pass']:>3}/{r['n']:<3}     vllm bnb4")
    else:
        print(f"{tag:<14}MISSING        -          vllm bnb4")
PY

log "DONE — results in $RESULTS, orchestrator log $LOGS/lcb55_vllm_4bit_orchestrator.log"
