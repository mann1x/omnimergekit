#!/bin/bash
# Phase 1 evals (mbpp + humaneval) on v4-MLP-passthrough Q6_K.
# Server expected to already be running on :8099 with --reasoning-budget 8192.
# Same chat-completions + apply_chat_template path as pod_omnimerge_v3_eval.sh
# (deepseek extraction strips <think> from scorer's view).
set -uo pipefail

WORKSPACE=/workspace
TOKENIZER="$WORKSPACE/base/qwen3.6-27b"
MODEL_NAME="Qwen3.6-27B-Omnimerge-v4-MLP"

CACHE_DIR="$WORKSPACE/eval_cache_v4mlp"
RESULTS_DIR="$WORKSPACE/eval_results_v4mlp"
LOG="$WORKSPACE/logs/eval_v4mlp.log"

mkdir -p "$CACHE_DIR" "$RESULTS_DIR" "$WORKSPACE/logs"

export PYTHONUNBUFFERED=1
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

run_eval() {
    local task="$1" max_gen="$2"
    echo "" | tee -a "$LOG"
    echo "============ $(date -Iseconds): $MODEL_NAME :: $task ============" | tee -a "$LOG"

    local model_args="model=$MODEL_NAME,base_url=http://localhost:8099/v1/chat/completions,num_concurrent=1,tokenizer_backend=huggingface,tokenizer=$TOKENIZER,max_length=32768"
    if [ -n "$max_gen" ]; then
        model_args="$model_args,max_gen_toks=$max_gen"
    fi

    PYTHONUNBUFFERED=1 lm_eval \
        --model local-chat-completions \
        --model_args "$model_args" \
        --tasks "$task" \
        --apply_chat_template \
        --confirm_run_unsafe_code \
        --batch_size 1 \
        --use_cache "$CACHE_DIR/${task}_${MODEL_NAME}" \
        --log_samples \
        --output_path "$RESULTS_DIR/${task}_${MODEL_NAME}" \
        2>&1 | tee -a "$LOG"
    echo "============ $(date -Iseconds): $MODEL_NAME :: $task DONE ============" | tee -a "$LOG"
}

echo "=== v4-MLP Phase 1 evals starting $(date -Iseconds) ===" | tee "$LOG"
echo "model: $MODEL_NAME (Q6_K)" | tee -a "$LOG"

# verify server up
if ! curl -s http://localhost:8099/health 2>/dev/null | grep -q '"status":"ok"'; then
    echo "!!! server not ready on :8099, aborting" | tee -a "$LOG"
    exit 1
fi
echo "    server reachable on :8099" | tee -a "$LOG"

run_eval mbpp ""
run_eval humaneval ""

echo "=== v4-MLP Phase 1 done $(date -Iseconds) ===" | tee -a "$LOG"
ls -la "$RESULTS_DIR/" | tee -a "$LOG"
