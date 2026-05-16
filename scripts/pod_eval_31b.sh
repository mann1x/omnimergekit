#!/bin/bash
# pod_eval_31b.sh — full canonical eval suite for Gemma 4 31B base + he1-it
# NVFP4A16 quants on a 2×3090 pod. Parallel across the 2 variants (one per GPU),
# sequential across the 9 templates per variant.
#
# CANONICAL LOCATION: omnimergekit/scripts/pod_eval_31b.sh
# Sibling to pod_reeval_failures.sh; runs the FULL chain (not the failure
# subset). Result schema is symmetric with the 98e-v4 + 128e suite runs in
# eval_results_vllm_suite/ — for cross-model comparison.
#
# Pre-reqs:
#   - /workspace/miniconda/envs/{omnimergekit,vllm}                (pod_setup_eval_envs.sh)
#   - /workspace/models/Gemma-4-31B-it-NVFP4A16/                   (pod_quant_31b.sh,
#                                                                   incl. preprocessor_config.json)
#   - /workspace/models/gemma-4-31b-he1-it-NVFP4A16/               (same)
#   - /workspace/omnimergekit/eval/templates/{template}.yaml       (canonical)
#
# Known gotchas baked in (same as pod_reeval_failures.sh, EVAL_PROTOCOL.md §4.4):
#   * `vllm` env for the server, `omnimergekit` env for omk_eval.
#   * --max-num-batched-tokens 8192 (Gemma 4 MM-encoder budget).
#   * --max-model-len 65536 — templates ask for max_gen_toks ≤ 49152; needs
#     prompt headroom to avoid HTTP 400.
#   * PATH=...envs/omnimergekit/bin:$PATH wraps omk_eval so its
#     subprocess.call(["lm-eval", ...]) resolves.
#   * preprocessor_config.json must exist in each model dir (vLLM 0.20.2 wheel
#     hard-requires it for Gemma4ForConditionalGeneration).
#
# Runtime estimate (per variant, 2-conc, NVFP4A16 31B dense on 3090 ≈ 80 tok/s):
#   * gpqa_diamond_full (198q) ~3-4 h
#   * arc_challenge_full (1172q, short) ~1 h
#   * humaneval/humanevalplus_full (164q each) ~30 min
#   * lcb_medium_55 (55q, long) ~90 min
#   * remaining 4 small (gsm8k_100/math500_100/aime_30/ifeval_100) ~1 h total
#   * TOTAL per variant: ~7-10 h
# Wall (parallel 2-GPU): ~7-10 h.
set -euo pipefail

MODELS=/workspace/models
OMK=/workspace/omnimergekit
RESULTS=/workspace/eval_results_31b
LOGS=/workspace/logs
mkdir -p "$RESULTS" "$LOGS"

# Defensive: ensure preprocessor_config.json exists in each model dir. vLLM
# 0.20.2 wheel hard-requires it for Gemma4ForConditionalGeneration. We synth
# it from processor_config.feature_extractor (HF transformers does NOT write
# preprocessor_config.json automatically for Gemma 4).
for D in "$MODELS/Gemma-4-31B-it-NVFP4A16" "$MODELS/gemma-4-31b-he1-it-NVFP4A16"; do
    [ -d "$D" ] || continue
    [ -f "$D/preprocessor_config.json" ] && continue
    PC=""
    for cand in "$D/processor_config.json" \
                "$MODELS/gemma-4-31B-it/processor_config.json" \
                "$MODELS/gemma-4-31b-he1-it/processor_config.json"; do
        [ -f "$cand" ] && PC="$cand" && break
    done
    # HF Hub fallback: if BF16 sources were purged by the quant pipeline
    # (pod_quant_31b.sh deletes them after a successful push) and the
    # NVFP4A16 dir wasn't authored with processor_config.json, pull from
    # the public base model. Hit on pod 36755693 2026-05-15. See
    # memory/feedback_gemma4_preprocessor_config_synth.md.
    if [ -z "$PC" ]; then
        TOKEN=$(cat /root/.cache/huggingface/token 2>/dev/null || echo "")
        for SLUG in google/gemma-4-31B-it google/gemma-4-31b-he1-it; do
            URL="https://huggingface.co/$SLUG/resolve/main/processor_config.json"
            if curl -fsSL -o "$D/processor_config.json" \
                 ${TOKEN:+-H "Authorization: Bearer $TOKEN"} "$URL" 2>/dev/null; then
                PC="$D/processor_config.json"
                echo "+ pulled processor_config.json from $SLUG → $D"
                break
            fi
        done
    fi
    if [ -n "$PC" ]; then
        /workspace/miniconda/envs/omnimergekit/bin/python - <<PY
import json
from pathlib import Path
proc = json.loads(Path("$PC").read_text())
fe = proc.get("feature_extractor", {})
if not fe:
    fe = proc  # whole-proc fallback when feature_extractor section is empty
Path("$D/preprocessor_config.json").write_text(json.dumps(fe, indent=2))
print(f"+ synthesized $D/preprocessor_config.json from $PC")
PY
    else
        echo "ERROR: no processor_config.json reachable for $D (HF Hub fallback also failed)."
        echo "       vLLM WILL fail to boot — fix manually before re-running."
    fi
done

# Canonical 9-template chain (same as scripts/eval_suite_vllm.sh on solidPC for v4)
TEMPLATES=(
    gpqa_diamond_full
    gsm8k_100
    math500_100
    aime_30
    arc_challenge_full
    ifeval_100
    humaneval_full
    humanevalplus_full
    lcb_medium_55
)

run_variant() {
    local NAME="$1" GPU="$2" PORT="$3" MODEL_DIR="$4" SERVED="$5"
    local OUT="$RESULTS/$NAME"
    local LOG="$LOGS/eval_${NAME}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$OUT"

    echo "[$NAME] GPU=$GPU port=$PORT model=$MODEL_DIR served=$SERVED log=$LOG"
    {
        echo "=== $NAME 31B-NVFP4A16 eval @ $(date) ==="
        # NOTE: tried PYTHONPATH overlay of /workspace/vllm-source for the
        # 3 Gemma 4 patches but the source dev build's .so files have a
        # torch ABI mismatch with the wheel. If 31B variants show silent-
        # empty completions on gsm8k/HE/MBPP at >10%, rebuild vllm from
        # source on the pod (`cd /workspace/vllm-source && pip install -e .`,
        # ~30-60 min on 3090) before trusting scores.
        #
        # MEMORY CONSTRAINT (2026-05-15): 31B NVFP4A16 (~15 GB weights) +
        # 65k-ctx KV cache (~12 GB on dense 31B) > 24 GB 3090. The 26B-A4B
        # parallel-1-GPU-per-variant config does NOT fit dense 31B.
        # Solution: tensor-parallel-size 2 (both GPUs per model, sequential
        # variants — see pod_eval_31b.sh outer block). Each model now sees
        # ~48 GB combined VRAM and fits cleanly.
        # See memory/feedback_31b_needs_tp2.md.
        CUDA_VISIBLE_DEVICES=$GPU \
        /workspace/miniconda/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_DIR" \
            --served-model-name "$SERVED" \
            --port "$PORT" \
            --tensor-parallel-size 2 \
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

        kill "$VLLM_PID" 2>/dev/null
        wait "$VLLM_PID" 2>/dev/null
        # Free both GPUs cleanly for the next variant (TP=2 leaves shards on both)
        sleep 3
        echo "=== $NAME eval DONE @ $(date) ==="
    }
}

# 31B dense needs TP=2 (both GPUs per variant). Variants run SEQUENTIALLY —
# parallel doesn't work because each variant needs both GPUs. Doubles
# wall-clock vs the 26B parallel chain (~14-20 h sequential vs ~7-10 h).
# Both variants use GPUs 0+1 via CUDA_VISIBLE_DEVICES=0,1.
run_variant 31b_nvfp4a16      0,1 8195 "$MODELS/Gemma-4-31B-it-NVFP4A16"      31b_nvfp4a16
run_variant 31b_he1_nvfp4a16  0,1 8196 "$MODELS/gemma-4-31b-he1-it-NVFP4A16"  31b_he1_nvfp4a16

echo "ALL 31B EVAL DONE @ $(date)"
