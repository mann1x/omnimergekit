#!/bin/bash
# Single-threaded sequential eval for MicroCoder candidate selection.
# Models:
#   - jackrong-v2 (LCB-30 ONLY; HE/MBPP already valid in eval_results)
#   - continuum-128k-forged (HE+MBPP+LCB-30; mrope_section fixed)
#   - jackrong-python      (HE+MBPP+LCB-30; lm_eval UnboundLocal patched)
#   - massivdash-ts        (HE+MBPP+LCB-30)
#
# Note: base Qwen3.5-4B is already done (HE 60.37 / MBPP 46.00 / LCB 3.33).
#
# No queueing, no parallel — strictly one model at a time. Server lifecycle
# is owned by THIS script only; nothing else may touch port 8099.
set -uo pipefail

WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
LLAMA=/opt/llama.cpp/build/bin
PYBIN=/shared/dev/lightseek/.venv/bin/python
TOK_BASE=$WS/hf_models_4b/Qwen3.5-4B
GGUF_DIR=$WS/4b_phase1/gguf_coder
GGUF_SOURCES=$WS/4b_phase1/gguf_sources
EVAL_DIR=$WS/4b_phase1/eval_results
LOGS=$WS/logs
mkdir -p "$EVAL_DIR" "$LOGS"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1

# ---- continuum mrope-fix: regenerate Q6_K if missing ----
ensure_continuum_q6k() {
    local Q6K=$GGUF_DIR/continuum-128k-forged.Q6_K.gguf
    [ -f "$Q6K" ] && return 0
    local SRC=$WS/hf_models_4b/coder_eval/continuum-128k-forged
    local F16=$GGUF_DIR/continuum-128k-forged.F16.gguf
    echo "[continuum] convert+quant"
    python3 /opt/llama.cpp/convert_hf_to_gguf.py "$SRC" --outtype f16 --outfile "$F16" 2>&1 | tail -3
    "$LLAMA/llama-quantize" "$F16" "$Q6K" Q6_K 2>&1 | tail -3
    rm -f "$F16"
}

start_server() {
    local GGUF=$1 NAME=$2
    pkill -f 'llama-server.*--port 8099' 2>/dev/null
    sleep 3
    nohup "$LLAMA/llama-server" \
        -m "$GGUF" --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
        --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 \
        > "$LOGS/4b_server_${NAME}.log" 2>&1 &
    SERVER_PID=$!
    disown
    for i in $(seq 1 60); do curl -sf http://localhost:8099/health >/dev/null 2>&1 && break; sleep 2; done
    curl -sf http://localhost:8099/health >/dev/null 2>&1 || {
        echo "[!] $NAME server failed"; tail -30 "$LOGS/4b_server_${NAME}.log"
        kill $SERVER_PID 2>/dev/null; return 1
    }
    echo "  server ready: $NAME"
}

stop_server() {
    [ -n "${SERVER_PID:-}" ] && kill $SERVER_PID 2>/dev/null
    SERVER_PID=
    sleep 2
}

run_he_mbpp() {
    local NAME=$1
    for task in mbpp humaneval; do
        local OP=$EVAL_DIR/${task}_${NAME}
        if [ -f "$OP/${NAME}/results_"*".json" ] 2>/dev/null; then
            echo "  $task already valid — skip"
            continue
        fi
        echo "  $task $(date -Iseconds)"
        mkdir -p "$OP"
        "$PYBIN" -u -m lm_eval \
            --model local-completions \
            --model_args "model=${NAME},base_url=http://localhost:8099/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=${TOK_BASE},max_gen_toks=2048" \
            --tasks "$task" \
            --gen_kwargs "temperature=0.0,top_p=1.0" \
            --batch_size 1 --use_cache "$OP/cache" \
            --log_samples --confirm_run_unsafe_code \
            --output_path "$OP" 2>&1 | tee "$LOGS/4b_${task}_${NAME}.log" | tail -8
    done
}

run_lcb() {
    local NAME=$1
    local LP=$EVAL_DIR/lcb_${NAME}
    if [ -f "$LP/lcb_results.json" ]; then
        local PA1=$(python3 -c "import json; print(json.load(open('$LP/lcb_results.json'))['pass_at_1'])" 2>/dev/null)
        if [ -n "$PA1" ] && [ "$PA1" != "0.0" ]; then
            echo "  lcb already valid: pass@1=$PA1 — skip"
            return 0
        fi
        rm -f "$LP/lcb_results.json" "$LP/lcb_samples.jsonl"
    fi
    mkdir -p "$LP"
    "$PYBIN" -u "$WS/scripts/lcb_llama_server.py" \
        --name "$NAME" --base-url http://localhost:8099 \
        --limit 30 --difficulty medium --min-date 2024-10-01 \
        --max-tokens 8192 \
        --output "$LP/lcb_results.json" \
        --samples "$LP/lcb_samples.jsonl" \
        2>&1 | tee "$LOGS/4b_lcb_${NAME}.log" | grep -E "PASS|FAIL|pass@1|Error" | tail -35
}

ensure_continuum_q6k

declare -a MODELS=(
    "qwen3.5-4b-base:$WS/4b_phase1/gguf_base/Qwen_Qwen3.5-4B-Q6_K.gguf:lcb_only"
    "jackrong-v2:$GGUF_SOURCES/jackrong-v2.Q6_K.gguf:lcb_only"
    "continuum-128k-forged:$GGUF_DIR/continuum-128k-forged.Q6_K.gguf:full"
    "jackrong-python:$GGUF_DIR/jackrong-python.Q6_K.gguf:full"
    "massivdash-ts:$GGUF_DIR/massivdash-ts.Q6_K.gguf:full"
)

for entry in "${MODELS[@]}"; do
    IFS=':' read -r NAME GGUF MODE <<< "$entry"
    [ -f "$GGUF" ] || { echo "[!] missing $GGUF — skip"; continue; }
    echo ""
    echo "========================================"
    echo "=== $NAME [$MODE] starting $(date -Iseconds) ==="
    echo "========================================"
    start_server "$GGUF" "$NAME" || continue
    if [ "$MODE" = "full" ]; then
        run_he_mbpp "$NAME"
    fi
    run_lcb "$NAME"
    stop_server
    echo "=== $NAME done $(date -Iseconds) ==="
done

echo ""
echo "=== all microcoder evals done $(date -Iseconds) ==="
