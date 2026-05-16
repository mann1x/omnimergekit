#!/bin/bash
# Relaunch vLLM with NO reasoning parser per arc template's design intent.
# Then re-run arc_challenge_reeval24k for both variants.
set -uo pipefail

MODELS=/workspace/models
OMK=/workspace/omnimergekit
RESULTS=/workspace/eval_results_reeval24k
LOGS=/workspace/logs

# Purge stale arc caches first
for V in 128e_nvfp4a16 98e_v4_nvfp4a16; do
    rm -rf "$RESULTS/$V/arc_challenge_reeval24k"
done

run_variant() {
    local NAME="$1" GPU="$2" PORT="$3" MODEL_DIR="$4" SERVED="$5"
    local OUT="$RESULTS/$NAME"
    local LOG="$LOGS/arc_noparser_${NAME}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$OUT"
    echo "[$NAME] GPU=$GPU log=$LOG"
    {
        echo "=== $NAME arc no-parser @ $(date) ==="
        # KEY DIFFERENCE: no --reasoning-parser, no default-chat-template-kwargs
        CUDA_VISIBLE_DEVICES=$GPU \
        /workspace/miniconda/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_DIR" --served-model-name "$SERVED" --port "$PORT" \
            --gpu-memory-utilization 0.92 --max-model-len 65536 \
            --max-num-batched-tokens 8192 --dtype bfloat16 --trust-remote-code \
            >> "$LOG" 2>&1 &
        VLLM_PID=$!
        for i in $(seq 1 60); do
            sleep 5
            curl -fs "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "[$NAME] ready @ ${i}*5s"; break; }
            kill -0 "$VLLM_PID" 2>/dev/null || { echo "[$NAME] FAIL boot"; exit 1; }
        done
        echo "[$NAME][arc] $(date +%H:%M:%S) start"
        PATH=/workspace/miniconda/envs/omnimergekit/bin:$PATH \
        PYTHONDONTWRITEBYTECODE=1 \
        /workspace/miniconda/envs/omnimergekit/bin/python "$OMK/eval/omk_eval.py" \
            --model "$MODEL_DIR" --template arc_challenge_reeval24k --backend vllm --no-server \
            --port "$PORT" --served-name "$SERVED" --tokenizer "$MODEL_DIR" \
            --results-dir "$OUT" >> "$LOG" 2>&1 \
            && echo "[$NAME][arc] $(date +%H:%M:%S) DONE" \
            || echo "[$NAME][arc] $(date +%H:%M:%S) FAIL"
        kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null
        echo "=== $NAME DONE @ $(date) ==="
    } &
}

run_variant 128e_nvfp4a16    0 8195 "$MODELS/Gemma-4-26B-A4B-it-NVFP4A16"  128e_nvfp4a16
run_variant 98e_v4_nvfp4a16  1 8196 "$MODELS/gemma-4-A4B-98e-v4-NVFP4A16"  98e_v4_nvfp4a16
wait
echo "ALL ARC NO-PARSER DONE @ $(date)"
