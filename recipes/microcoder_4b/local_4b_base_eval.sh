#!/bin/bash
# Base Qwen3.5-4B eval — HE + MBPP, same Q6_K methodology as M1..M5.
# Uses Bartowski's pre-built Q6_K GGUF.
set -uo pipefail

WORKSPACE=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
LLAMA=/opt/llama.cpp/build/bin
GGUF="$WORKSPACE/4b_phase1/gguf_base/Qwen_Qwen3.5-4B-Q6_K.gguf"
TOK="$WORKSPACE/hf_models_4b/Qwen3.5-4B"
NAME="Qwen3.5-4B-base"
EVAL_DIR="$WORKSPACE/4b_phase1/eval_results"
LOGS="$WORKSPACE/logs"
mkdir -p "$EVAL_DIR" "$LOGS"

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1

echo "=== Base Qwen3.5-4B eval $(date -Iseconds) ==="

pkill -f 'llama-server.*--port 8099' 2>/dev/null
sleep 2
nohup "$LLAMA/llama-server" \
    -m "$GGUF" --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
    --reasoning-format deepseek --reasoning-budget 8192 \
    --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 \
    > "$LOGS/4b_server_base.log" 2>&1 &
SERVER_PID=$!
disown
for i in $(seq 1 60); do
    curl -sf http://localhost:8099/health >/dev/null 2>&1 && break
    sleep 2
done
curl -sf http://localhost:8099/health >/dev/null 2>&1 || { echo "[!] server failed"; tail -20 "$LOGS/4b_server_base.log"; exit 1; }
echo "    server ready (PID $SERVER_PID)"

for task in mbpp humaneval; do
    echo "=== $task $(date -Iseconds) ==="
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
        2>&1 | tee "$LOGS/4b_${task}_base.log" | tail -25
done

kill "$SERVER_PID" 2>/dev/null
sleep 2
echo "=== done $(date -Iseconds) ==="
find "$EVAL_DIR" -path "*${NAME}*" -name "results_*.json"
