#!/usr/bin/env bash
# LCB-medium 55q re-eval on the LOCAL RTX 3090 via vLLM 4-bit (bitsandbytes
# on-load) for both 128e and 98e-v4-it bf16 sources. Compares vLLM-served
# numbers against the published llama.cpp Q6_K numbers to decide whether the
# eval protocol should switch to vLLM.
#
# Triggered AFTER the two currently in-flight runs finish:
#   - pod he1-it Q4_K_M HE/MBPP (tmux he1eval on ssh3.vast.ai:10024)
#   - local 128e LCB-medium @16k (pid passed as $1; default 1006141)
#
# Args:
#   $1 — pid to wait on locally (default 1006141 — the 128e LCB-16k runner)
#   $2 — set to "skip-pod-wait" to NOT wait on the pod tmux session
#
# Outputs:
#   eval_results_vllm4bit/lcb_med_55q/{128e,v4}/...
#   logs/vllm_4bit_lcb_*.log
#
# Notes:
#   - bf16 on-load with `--quantization bitsandbytes --load-format bitsandbytes`
#     re-quantizes weights at server startup. nf4 on 26B-A4B ≈ 13-14 GB.
#   - --enforce-eager is required because Gemma 4 head_dim=512 isn't supported
#     by flash-attn in vLLM 0.19.0.
#   - LCB runner is the SAME OpenAI-compatible client (eval/lcb/lcb_llama_server.py)
#     used for the llama.cpp runs, so any apples-to-apples delta is purely
#     serving-stack.

set -euo pipefail

WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
OMK=/shared/dev/omnimergekit
LOGS=$WS/logs
RESULTS=$WS/eval_results_vllm4bit/lcb_med_55q
mkdir -p "$LOGS" "$RESULTS"

PRED_PID=${1:-1006141}
SKIP_POD=${2:-}
POD_SSH="ssh -o ConnectTimeout=10 -o ServerAliveInterval=20 -p 10024 root@ssh3.vast.ai"

ts() { date +%Y%m%d_%H%M%S; }
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOGS/vllm_4bit_lcb_orchestrator.log"; }

# ---------- 1. wait for local 128e LCB-16k runner to finish ----------
log "waiting for local LCB-16k runner pid=$PRED_PID ..."
while kill -0 "$PRED_PID" 2>/dev/null; do sleep 60; done
log "  local runner exited"

# ---------- 2. wait for pod he1 eval (tmux he1eval) ----------
if [[ "$SKIP_POD" != "skip-pod-wait" ]]; then
    log "waiting for pod he1 eval (tmux he1eval on ssh3.vast.ai:10024) ..."
    while $POD_SSH 'tmux has-session -t he1eval 2>/dev/null' 2>/dev/null; do
        sleep 120
    done
    log "  pod he1 eval finished (tmux gone)"
fi

# ---------- 3. ensure 98e-v4 bf16 weights are local ----------
V4_LOCAL=$WS/google/gemma-4-A4B-98e-v4-it
if [[ ! -f "$V4_LOCAL/model.safetensors.index.json" && ! -f "$V4_LOCAL/model-00001-of-00021.safetensors" ]]; then
    log "downloading 98e-v4-it bf16 weights to $V4_LOCAL ..."
    hf download ManniX-ITA/gemma-4-A4B-98e-v4-it --local-dir "$V4_LOCAL" \
        --exclude "*.gguf" "imatrix*" 2>&1 | tail -20 | tee -a "$LOGS/vllm_4bit_lcb_orchestrator.log"
else
    log "98e-v4-it bf16 already local at $V4_LOCAL"
fi

# ---------- 4. clean GPU before launching ----------
pkill -KILL -f "llama-server.*--port (8099|8089|8090)" 2>/dev/null || true
sleep 3

# ---------- 5. run LCB on each model ----------
run_one () {
    local NAME="$1"      # short tag for output files
    local MODEL_DIR="$2" # local bf16 path
    local PORT=8090
    local SDIR="$RESULTS/$NAME"
    mkdir -p "$SDIR"

    local SLOG="$LOGS/vllm_4bit_lcb_${NAME}_$(ts).log"
    log "=== launching vLLM 4-bit for $NAME (model=$MODEL_DIR) ==="

    /root/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_DIR" \
        --served-model-name "$NAME" \
        --quantization bitsandbytes \
        --load-format bitsandbytes \
        --port $PORT \
        --gpu-memory-utilization 0.92 \
        --max-model-len 32768 \
        --enforce-eager \
        --dtype bfloat16 \
        > "$SLOG" 2>&1 &
    local SPID=$!
    disown

    # readiness loop (vLLM takes ~3-5 min to bnb-requantize 26B on first load)
    log "  vllm pid=$SPID slog=$SLOG; waiting up to 12 min for ready ..."
    local ready=0
    for i in $(seq 1 360); do
        if curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
            log "  vLLM ready after ${i}×2s"
            ready=1
            break
        fi
        if ! kill -0 "$SPID" 2>/dev/null; then
            log "  ABORT: vllm pid=$SPID died during startup"
            tail -40 "$SLOG" | sed 's/^/    /'
            return 2
        fi
        sleep 2
    done
    if [[ $ready -ne 1 ]]; then
        log "  ABORT: vllm never responded; tail of $SLOG:"
        tail -40 "$SLOG" | sed 's/^/    /'
        kill -KILL $SPID 2>/dev/null || true
        return 3
    fi

    # smoke test: 1 small completion so we know the actual model speaks
    log "  smoke test on /v1/chat/completions ..."
    curl -sS "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Write Python: def add(a,b): return\"}],\"max_tokens\":32,\"temperature\":0}" \
        | tee -a "$SDIR/smoke.json"
    echo >> "$SDIR/smoke.json"

    # run LCB-medium 55q (same OpenAI-compatible runner)
    log "  running LCB-medium 55q on $NAME via vLLM ..."
    local T0=$(date +%s)
    /root/anaconda3/envs/omnimergekit/bin/python "$OMK/eval/lcb/lcb_llama_server.py" \
        --name "$NAME" --base-url "http://localhost:$PORT" \
        --limit 999 --max-tokens 16384 \
        --output "$SDIR/${NAME}_lcb_med_55q_vllm4bit.json" \
        > "$LOGS/vllm_4bit_lcb_${NAME}_runner_$(ts).log" 2>&1 || true
    log "  LCB wall=$(($(date +%s)-T0))s"

    # cleanup
    kill -TERM $SPID 2>/dev/null || true
    sleep 5
    kill -KILL $SPID 2>/dev/null || true
    pkill -KILL -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null || true
    sleep 5

    # quick result summary
    python3 - <<PY | tee -a "$LOGS/vllm_4bit_lcb_orchestrator.log"
import json
try:
    r = json.load(open("$SDIR/${NAME}_lcb_med_55q_vllm4bit.json"))
    print(f"  RESULT [$NAME] vllm-bnb4bit LCB-medium pass@1 = {r['pass_at_1']*100:.2f}%  ({r['n_pass']}/{r['n']})")
except Exception as e:
    print(f"  RESULT [$NAME] ERROR: {e}")
PY
}

run_one "128e_bnb4" "$WS/google/gemma-4-26B-A4B-it"
run_one "v4_bnb4"   "$V4_LOCAL"

# ---------- 6. side-by-side report ----------
log "=== FINAL SIDE-BY-SIDE ==="
python3 - <<'PY' | tee -a "$LOGS/vllm_4bit_lcb_orchestrator.log"
import json, pathlib
root = pathlib.Path("/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models")
table = []
# llama.cpp baselines (from published numbers)
llamacpp = {
    "128e": ("eval_results_128e_full_v2/lcb/gemma4-A4B-128e-Q6K_lcb_full.json", 0.8727, 55),
    "v4":   ("eval_results_v4/lcb_16k/gemma4-A4B-98e-v4-it-Q6K_lcb_full_16k.json", 0.7818, 55),
}
for tag, (p, ref, n) in llamacpp.items():
    try:
        r = json.load(open(root / p))
        table.append((tag, "llama.cpp Q6_K", f"{r['pass_at_1']*100:.2f}%", f"{r['n_pass']}/{r['n']}"))
    except Exception:
        table.append((tag, "llama.cpp Q6_K", f"{ref*100:.2f}% (cached)", f"?/{n}"))
# vllm bnb4
for tag, src in [("128e","128e_bnb4"), ("v4","v4_bnb4")]:
    p = root / f"eval_results_vllm4bit/lcb_med_55q/{src}/{src}_lcb_med_55q_vllm4bit.json"
    if p.exists():
        r = json.load(open(p))
        table.append((tag, "vllm bnb4", f"{r['pass_at_1']*100:.2f}%", f"{r['n_pass']}/{r['n']}"))
    else:
        table.append((tag, "vllm bnb4", "MISSING", "?"))

print(f"{'variant':<8}{'serving':<22}{'pass@1':<14}{'n_pass/n'}")
for row in table:
    print(f"{row[0]:<8}{row[1]:<22}{row[2]:<14}{row[3]}")
PY

log "DONE — see $RESULTS and the orchestrator log $LOGS/vllm_4bit_lcb_orchestrator.log"
