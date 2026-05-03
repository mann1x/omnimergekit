#!/bin/bash
# Omnimerge v3a + v3b evaluations on pod (mirrors pod_omnimerge_v2.sh)
# Order: phase 1 = MBPP+HumanEval for both (fast); phase 2 = GPQA Diamond for both (long)
set -uo pipefail   # NOT -e: a single eval failure must not kill subsequent ones (cache will resume on rerun)

WORKSPACE=/workspace
GGUF_V3A="$WORKSPACE/gguf/Qwen3.6-27B-Omnimerge-v3a-Q6_K.gguf"
GGUF_V3B="$WORKSPACE/gguf/Qwen3.6-27B-Omnimerge-v3b-Q6_K.gguf"
TOKENIZER="$WORKSPACE/base/qwen3.6-27b"

CACHE_DIR="$WORKSPACE/eval_cache_v3"
RESULTS_DIR="$WORKSPACE/eval_results_v3"
LOG="$WORKSPACE/logs/eval_v3.log"
SERVER_LOG="$WORKSPACE/logs/llama_server_v3.log"

mkdir -p "$CACHE_DIR" "$RESULTS_DIR" "$WORKSPACE/logs"

export PYTHONUNBUFFERED=1
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

# Bare `pip install lm-eval` does NOT pull tenacity/requests/aiohttp — local-completions
# fails at construction. Self-heal on rerun. Memo: memory/feedback_lm_eval_pod_deps.md
if ! python -c 'from lm_eval.models.openai_completions import LocalCompletionsAPI' 2>/dev/null; then
    echo ">>> installing lm-eval[api] extras..." | tee -a "$LOG"
    pip install -q 'lm-eval[api]' 2>&1 | tail -3 | tee -a "$LOG"
fi

LLAMA_BIN=/workspace/llama.cpp/build/bin/llama-server

start_server() {
    local gguf="$1" budget="$2"
    pkill -f '[l]lama-server' 2>/dev/null || true
    sleep 3
    echo ">>> launching llama-server: $(basename "$gguf") budget=$budget" | tee -a "$LOG"
    "$LLAMA_BIN" -m "$gguf" \
        --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
        --reasoning-format deepseek --reasoning-budget "$budget" \
        > "$SERVER_LOG" 2>&1 &
    disown
    for i in $(seq 1 60); do
        if curl -s http://localhost:8099/health 2>/dev/null | grep -q '"status":"ok"'; then
            echo "    server ready after ${i}s" | tee -a "$LOG"
            return 0
        fi
        sleep 2
    done
    echo "!!! server did not become ready in 120s" | tee -a "$LOG"
    return 1
}

run_eval() {
    local model_name="$1" task="$2" extra="$3" max_gen="$4"
    echo "" | tee -a "$LOG"
    echo "============ $(date -Iseconds): $model_name :: $task ============" | tee -a "$LOG"

    # /v1/chat/completions + apply_chat_template + reasoning-format deepseek
    # Reason: Qwen3.6 base auto-emits <think> on raw /v1/completions (saw 81% leak in v3a MBPP).
    # Chat endpoint with deepseek extraction strips reasoning into reasoning_content,
    # leaving only the final answer for lm-eval scorer. Memo: 2026-04-29 v3a methodology shift.
    local model_args="model=$model_name,base_url=http://localhost:8099/v1/chat/completions,num_concurrent=1,tokenizer_backend=huggingface,tokenizer=$TOKENIZER,max_length=32768"
    if [ -n "$max_gen" ]; then
        model_args="$model_args,max_gen_toks=$max_gen"
    fi

    PYTHONUNBUFFERED=1 lm_eval \
        --model local-chat-completions \
        --model_args "$model_args" \
        --tasks "$task" \
        --apply_chat_template \
        --batch_size 1 \
        $extra \
        --use_cache "$CACHE_DIR/${task}_${model_name}" \
        --log_samples \
        --output_path "$RESULTS_DIR/${task}_${model_name}" \
        2>&1 | tee -a "$LOG"
    echo "============ $(date -Iseconds): $model_name :: $task DONE ============" | tee -a "$LOG"
}

stop_server() {
    pkill -f '[l]lama-server' 2>/dev/null || true
    sleep 3
}

echo "=== Omnimerge v3 evals starting $(date -Iseconds) ===" | tee "$LOG"
echo "v3a: $GGUF_V3A" | tee -a "$LOG"
echo "v3b: $GGUF_V3B" | tee -a "$LOG"

#######################################
# Phase 1: MBPP + HumanEval (both models, budget 8192)
#######################################
for spec in "v3a:$GGUF_V3A" "v3b:$GGUF_V3B"; do
    NAME="${spec%%:*}"
    GGUF="${spec##*:}"
    MODEL_NAME="Qwen3.6-27B-Omnimerge-$NAME"
    start_server "$GGUF" 8192 || { echo "skip $NAME phase1" | tee -a "$LOG"; continue; }
    run_eval "$MODEL_NAME" mbpp "--confirm_run_unsafe_code"  ""
    run_eval "$MODEL_NAME" humaneval "--confirm_run_unsafe_code" ""
    stop_server
done

#######################################
# Phase 2: GPQA Diamond (both models, budget 16384, chat template)
#######################################
for spec in "v3a:$GGUF_V3A" "v3b:$GGUF_V3B"; do
    NAME="${spec%%:*}"
    GGUF="${spec##*:}"
    MODEL_NAME="Qwen3.6-27B-Omnimerge-$NAME"
    start_server "$GGUF" 16384 || { echo "skip $NAME phase2" | tee -a "$LOG"; continue; }
    run_eval "$MODEL_NAME" gpqa_diamond_cot_zeroshot "" 16384
    stop_server
done

echo "=== ALL EVALS DONE $(date -Iseconds) ===" | tee -a "$LOG"
ls -la "$RESULTS_DIR/" | tee -a "$LOG"
