#!/bin/bash
# pod_reeval_fix_a.sh — re-run arc/gsm8k/math500 reeval24k after applying
# lm-eval Fix-A (reasoning_content fallback). Parallel across 2 variants
# (one GPU each), sequential across the 3 templates per variant.
#
# Run inside tmux session `fix_a` for ssh-disconnect safety.
set -uo pipefail   # NOTE: no -e — keep going on per-template FAIL

MODELS=/workspace/models
OMK=/workspace/omnimergekit
RESULTS=/workspace/eval_results_reeval24k
LOGS=/workspace/logs
mkdir -p "$RESULTS" "$LOGS"

TEMPLATES=(arc_challenge_reeval24k gsm8k_reeval24k math500_reeval24k)

run_variant() {
    local NAME="$1" GPU="$2" PORT="$3" MODEL_DIR="$4" SERVED="$5"
    local OUT="$RESULTS/$NAME"
    local LOG="$LOGS/fix_a_${NAME}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$OUT"
    echo "[$NAME] GPU=$GPU port=$PORT served=$SERVED log=$LOG"
    {
        echo "=== $NAME fix-A reeval @ $(date) ==="
        CUDA_VISIBLE_DEVICES=$GPU \
        /workspace/miniconda/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_DIR" --served-model-name "$SERVED" --port "$PORT" \
            --gpu-memory-utilization 0.92 --max-model-len 65536 \
            --max-num-batched-tokens 8192 --dtype bfloat16 --trust-remote-code \
            --reasoning-parser gemma4 \
            --default-chat-template-kwargs '{"enable_thinking": true}' \
            >> "$LOG" 2>&1 &
        VLLM_PID=$!
        echo "[$NAME] vLLM pid=$VLLM_PID"
        for i in $(seq 1 60); do
            sleep 5
            curl -fs "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "[$NAME] ready @ ${i}*5s"; break; }
            kill -0 "$VLLM_PID" 2>/dev/null || { echo "[$NAME] FAIL boot"; exit 1; }
        done
        for TPL in "${TEMPLATES[@]}"; do
            echo "[$NAME][$TPL] $(date +%H:%M:%S) start"
            PATH=/workspace/miniconda/envs/omnimergekit/bin:$PATH \
            PYTHONDONTWRITEBYTECODE=1 \
            /workspace/miniconda/envs/omnimergekit/bin/python "$OMK/eval/omk_eval.py" \
                --model "$MODEL_DIR" --template "$TPL" --backend vllm --no-server \
                --port "$PORT" --served-name "$SERVED" --tokenizer "$MODEL_DIR" \
                --results-dir "$OUT" >> "$LOG" 2>&1 \
                && echo "[$NAME][$TPL] $(date +%H:%M:%S) DONE" \
                || echo "[$NAME][$TPL] $(date +%H:%M:%S) FAIL"
        done
        kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null; sleep 3
        echo "=== $NAME DONE @ $(date) ==="
    } &
}

run_variant 128e_nvfp4a16    0 8195 "$MODELS/Gemma-4-26B-A4B-it-NVFP4A16"  128e_nvfp4a16
run_variant 98e_v4_nvfp4a16  1 8196 "$MODELS/gemma-4-A4B-98e-v4-NVFP4A16"  98e_v4_nvfp4a16
wait
echo "ALL FIX-A REEVAL DONE @ $(date)"
