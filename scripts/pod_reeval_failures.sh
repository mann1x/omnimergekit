#!/bin/bash
# pod_reeval_failures.sh — symmetric re-eval on the pod, 2 GPUs parallel.
#
# CANONICAL LOCATION: omnimergekit/scripts/pod_reeval_failures.sh
# Any copy under backup_models/scripts/ or on a pod at /workspace/scripts/
# is a downstream rsync target; edits MUST land here first then be pushed.
#
# Runs the 8 reeval24k templates against 128e (GPU 0) and v4 (GPU 1) in
# parallel tmux sessions. Each template targets the failure-subset doc_ids
# from scripts/reeval_failure_manifest.json (206 problems total per model).
#
# Expected wall: 2-4 h depending on thinking depth × number of length-cap
# survivors. SQLite cache lives under each variant's results dir, so any
# kill/restart resumes safely.
#
# Pre-reqs (set up by pod_bootstrap_reeval.sh + pod_setup_eval_envs.sh +
# earlier rsyncs):
#   - /workspace/miniconda/envs/{omnimergekit,vllm} (see pod_setup_eval_envs.sh)
#   - /workspace/models/Gemma-4-26B-A4B-it-NVFP4A16  (vLLM-ready, incl.
#                                                    preprocessor_config.json)
#   - /workspace/models/gemma-4-A4B-98e-v4-NVFP4A16  (same)
#   - /workspace/omnimergekit/eval/templates/*reeval24k.yaml
#   - /workspace/omnimergekit/eval/lm_eval_tasks/reeval24k/*.yaml (INLINED)
#   - /workspace/scripts/reeval_failure_manifest.json
#
# Known gotchas baked in (each documented in eval/EVAL_PROTOCOL.md §4.4):
#   * Two conda envs are MANDATORY — `vllm` for the server, `omnimergekit`
#     for omk_eval. transformers versions diverge (5.5.0 vs vllm's pin),
#     never mix.
#   * --max-num-batched-tokens 8192 satisfies Gemma 4's MM-encoder budget
#     (default 2048 < max_tokens_per_mm_item 2496 → vLLM boot crash).
#   * --max-model-len 65536 (not 49152) — templates ask for max_gen_toks=49152
#     and vLLM rejects with HTTP 400 if prompt+output > max-model-len. 65536
#     gives ~16k headroom for system+user prompt.
#   * PATH=...envs/omnimergekit/bin:$PATH wraps the omk_eval call so its
#     subprocess.call(["lm-eval", ...]) resolves (we invoke python by full
#     path so PATH would otherwise miss the env's bin/).
set -euo pipefail

MODELS=/workspace/models
OMK=/workspace/omnimergekit
RESULTS=/workspace/eval_results_reeval24k
LOGS=/workspace/logs
mkdir -p "$RESULTS" "$LOGS"

# 8 failure-subset templates (no LCB — LCB tracked separately)
TEMPLATES=(
    gpqa_diamond_reeval24k
    gsm8k_reeval24k
    math500_reeval24k
    aime_reeval24k
    arc_challenge_reeval24k
    ifeval_reeval24k
    humaneval_reeval24k
    humanevalplus_reeval24k
)

run_variant() {
    local NAME="$1" GPU="$2" PORT="$3" MODEL_DIR="$4" SERVED="$5"
    local OUT="$RESULTS/$NAME"
    local LOG="$LOGS/reeval_${NAME}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$OUT"

    echo "[$NAME] GPU=$GPU port=$PORT model=$MODEL_DIR served=$SERVED log=$LOG"
    {
        echo "=== $NAME re-eval @ $(date) ==="
        # Launch vLLM on this GPU using the dedicated vllm conda env.
        # NOTE: PYTHONPATH overlay of /workspace/vllm-source was tried but the
        # source is on a much newer dev branch (`g630492da3`) and the .so
        # files clash with the wheel's torch ABI (undefined symbol _ZN3c10...).
        # If silent-empty rates exceed ~10% on gsm8k or HE, rebuild vllm from
        # source on the pod instead of overlaying.
        CUDA_VISIBLE_DEVICES=$GPU \
        /workspace/miniconda/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_DIR" \
            --served-model-name "$SERVED" \
            --port "$PORT" \
            --gpu-memory-utilization 0.92 \
            --max-model-len 65536 \
            --max-num-batched-tokens 8192 \
            --dtype bfloat16 \
            --trust-remote-code \
            --reasoning-parser gemma4 \
            --default-chat-template-kwargs '{"enable_thinking": true}' \
            >> "$LOG" 2>&1 &
        VLLM_PID=$!
        echo "[$NAME] vLLM pid=$VLLM_PID, waiting for /health"

        # Wait up to 5 min for vLLM ready
        for i in $(seq 1 60); do
            sleep 5
            if curl -fs "http://localhost:$PORT/health" >/dev/null 2>&1; then
                echo "[$NAME] vLLM ready after ${i}*5s"
                break
            fi
            if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                echo "[$NAME] FAIL: vLLM died during boot"
                exit 1
            fi
        done

        # Run each reeval24k template via omk_eval
        # omk_eval shells out to `lm-eval` via subprocess, so the omnimergekit
        # env's bin dir MUST be on PATH (not enough to invoke its python).
        for TPL in "${TEMPLATES[@]}"; do
            echo "[$NAME][$TPL] $(date +%H:%M:%S) start"
            PATH=/workspace/miniconda/envs/omnimergekit/bin:$PATH \
            /workspace/miniconda/envs/omnimergekit/bin/python "$OMK/eval/omk_eval.py" \
                --model "$MODEL_DIR" \
                --template "$TPL" \
                --backend vllm \
                --no-server \
                --port "$PORT" \
                --served-name "$SERVED" \
                --tokenizer "$MODEL_DIR" \
                --results-dir "$OUT" \
                >> "$LOG" 2>&1 \
                && echo "[$NAME][$TPL] $(date +%H:%M:%S) DONE" \
                || echo "[$NAME][$TPL] $(date +%H:%M:%S) FAIL"
        done

        # Stop vLLM
        kill "$VLLM_PID" 2>/dev/null
        wait "$VLLM_PID" 2>/dev/null
        echo "=== $NAME re-eval DONE @ $(date) ==="
    } &
}

# Launch in parallel tmux session sub-shells
run_variant 128e_nvfp4a16 0 8195 "$MODELS/Gemma-4-26B-A4B-it-NVFP4A16" 128e_nvfp4a16
run_variant 98e_v4_nvfp4a16 1 8196 "$MODELS/gemma-4-A4B-98e-v4-NVFP4A16" 98e_v4_nvfp4a16

wait
echo "ALL REEVAL DONE @ $(date)"
