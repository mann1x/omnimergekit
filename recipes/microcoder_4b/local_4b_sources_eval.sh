#!/bin/bash
# Eval the two M1..M5 source fine-tunes (jackrong-v2, crow-4b) on HE+MBPP at Q6_K.
# Same methodology as base + M1..M5: parallel-2, raw /v1/completions, max_gen_toks=2048.
set -uo pipefail

WORKSPACE=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
LLAMA=/opt/llama.cpp/build/bin
GGUF_DIR="$WORKSPACE/4b_phase1/gguf_sources"
EVAL_DIR="$WORKSPACE/4b_phase1/eval_results"
LOGS="$WORKSPACE/logs"
TOK_BASE="$WORKSPACE/hf_models_4b/Qwen3.5-4B"  # tokenizer compatible across all 3 (same family)
mkdir -p "$EVAL_DIR" "$LOGS"

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1

run_eval() {
    local NAME=$1
    local GGUF=$2
    local TOK=$3

    echo "=== $NAME starting $(date -Iseconds) ==="
    pkill -f 'llama-server.*--port 8099' 2>/dev/null
    sleep 2
    nohup "$LLAMA/llama-server" \
        -m "$GGUF" --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
        --reasoning-format deepseek --reasoning-budget 8192 \
        --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 \
        > "$LOGS/4b_server_${NAME}.log" 2>&1 &
    SERVER_PID=$!
    disown
    for i in $(seq 1 60); do
        curl -sf http://localhost:8099/health >/dev/null 2>&1 && break
        sleep 2
    done
    curl -sf http://localhost:8099/health >/dev/null 2>&1 || { echo "[!] server $NAME failed"; tail -20 "$LOGS/4b_server_${NAME}.log"; return 1; }
    echo "  server ready"

    for task in mbpp humaneval; do
        echo "  $task $(date -Iseconds)"
        OUTPATH="$EVAL_DIR/${task}_${NAME}"
        mkdir -p "$OUTPATH"
        /shared/dev/lightseek/.venv/bin/python -u -m lm_eval \
            --model local-completions \
            --model_args "model=${NAME},base_url=http://localhost:8099/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=${TOK},max_gen_toks=2048" \
            --tasks "$task" \
            --gen_kwargs "temperature=0.0,top_p=1.0" \
            --batch_size 1 --use_cache "$OUTPATH/cache" \
            --log_samples --confirm_run_unsafe_code \
            --output_path "$OUTPATH" \
            2>&1 | tee "$LOGS/4b_${task}_${NAME}.log" | tail -15
    done

    kill "$SERVER_PID" 2>/dev/null
    sleep 2
}

run_eval "jackrong-v2" "$GGUF_DIR/jackrong-v2.Q6_K.gguf" "$TOK_BASE"
run_eval "crow-4b" "$GGUF_DIR/crow-4b.Q6_K.gguf" "$TOK_BASE"

echo "=== all sources done $(date -Iseconds) ==="
find "$EVAL_DIR" -path "*jackrong-v2*" -o -path "*crow-4b*" 2>/dev/null | grep "results_.*json" | head
