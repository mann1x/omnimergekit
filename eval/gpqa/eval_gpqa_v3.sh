#!/bin/bash
# Canonical full GPQA Diamond eval script using LM-EVAL DIRECTLY.
# Multi-arch: works across model families (Gemma 4, Qwen3.5, DeepSeek, etc)
# via env-overridable sampling + tokenizer + reasoning budget.
#
# Usage:
#   ./scripts/eval_gpqa_v3.sh <NAME> <GGUF_PATH>
#
# Family preset (set before call, or pass on command line):
#   MODEL_FAMILY=gemma4     → temp 1.0 top_p 0.95 top_k 64 (default, Gemma 4 official)
#   MODEL_FAMILY=qwen3.5    → temp 0.6 top_p 0.95 top_k 20 (Qwen3 reasoning mode)
#   MODEL_FAMILY=qwen3-coder → temp 0.7 top_p 0.8  top_k 20
#   MODEL_FAMILY=deterministic → temp 0.0 top_p 1.0  top_k 0  (greedy; fair comparative)
#
# Optional env var overrides (override MODEL_FAMILY defaults if set):
#   PORT=8099                    llama-server port (use 8100 for parallel run on GPU 1)
#   CUDA_VISIBLE_DEVICES=0       which GPU to use
#   SEED=42                      sampling seed
#   TEMP                         sampling temperature
#   TOP_P                        top-p
#   TOP_K                        top-k
#   CTX=32768                    context size
#   REASONING_BUDGET=16384       max thinking tokens (deepseek format)
#   MAX_GEN_TOKS=24576           lm-eval gen_kwargs max output tokens
#   DRY_MULTIPLIER=0.5           DRY anti-repetition sampler
#   SAMPLES='{"task":[i,i,...]}' lm-eval --samples filter for partial/resume
#                                e.g. SAMPLES='{"gpqa_diamond_cot_zeroshot":[0]}' for 1q probe
#   CONDA_ENV=lightseek          conda env (lightseek on solidpc, base on pod)
#   TOKENIZER_PATH=...           tokenizer HF path (default: Gemma 4)
#   REPO_ROOT                    override output root (default: auto-detect from script)
#   LLAMA_BIN=/opt/llama.cpp/build/bin  llama.cpp binary dir
#
# LOCKED METHODOLOGY anchors (do not change):
#   --reasoning-format deepseek   chain-of-thought hidden in reasoning_content field
#   --apply_chat_template         server handles chat template via --jinja (implicit)
#   num_concurrent 1              deterministic, single-threaded eval
#   MAX_RETRIES=6                 retry loop for llama.cpp PEG parser crashes
#
# Notes:
#   - top_k and seed go via the SERVER CLI defaults (lm_eval doesn't pass them)
#   - The server CLI temp/top_p are mirrored in gen_kwargs as belt-and-suspenders
#   - Gemma 4 tokenizer is `google/gemma-4-26B-A4B-it`
#   - Qwen3.5 tokenizer: pass the HF snapshot dir or Hugging Face repo id directly
set -euo pipefail

NAME="${1:-}"
GGUF="${2:-}"

if [[ -z "$NAME" || -z "$GGUF" ]]; then
    cat <<'USAGE'
Usage: $0 <NAME> <GGUF_PATH>

Env (required for non-Gemma models):
  TOKENIZER_PATH=<hf_dir_or_repo>   tokenizer for this model

Env (optional):
  MODEL_FAMILY=gemma4|qwen3.5|qwen3-coder|deterministic
  TEMP, TOP_P, TOP_K, CTX, REASONING_BUDGET, MAX_GEN_TOKS, DRY_MULTIPLIER
  SAMPLES='{"gpqa_diamond_cot_zeroshot":[0]}'   # 1q probe
  PORT, SEED, CUDA_VISIBLE_DEVICES, CONDA_ENV, REPO_ROOT, LLAMA_BIN

Examples:
  # Gemma 4 (default)
  ./eval_gpqa_v3.sh gemma-4-128e google/gemma-4-A4B-128e-Q6_K.gguf

  # Qwen3.5 with tokenizer from local HF snapshot
  MODEL_FAMILY=qwen3.5 TOKENIZER_PATH=/path/to/qwen-hf-dir \
    ./eval_gpqa_v3.sh qwen3.5-omnimerge Qwen3.5-27B-Omnimerge-Q6_K.gguf

  # 1-question probe (resume_index 0)
  SAMPLES='{"gpqa_diamond_cot_zeroshot":[0]}' \
  MODEL_FAMILY=deterministic TOKENIZER_PATH=/path/to/tok \
    ./eval_gpqa_v3.sh probe Qwen3.5-27B-Omnimerge-Q6_K.gguf
USAGE
    exit 1
fi

if [[ ! -f "$GGUF" ]]; then
    echo "ERROR: GGUF not found: $GGUF"
    exit 1
fi

# --- Family presets (used only if TEMP/TOP_P/TOP_K not explicitly overridden) ---
MODEL_FAMILY="${MODEL_FAMILY:-gemma4}"
case "$MODEL_FAMILY" in
    gemma4)
        FAM_TEMP=1.0; FAM_TOP_P=0.95; FAM_TOP_K=64
        ;;
    qwen3.5|qwen3_5|qwen35)
        FAM_TEMP=0.6; FAM_TOP_P=0.95; FAM_TOP_K=20
        ;;
    qwen3-coder|qwen3_coder)
        FAM_TEMP=0.7; FAM_TOP_P=0.8; FAM_TOP_K=20
        ;;
    deterministic|greedy)
        FAM_TEMP=0.0; FAM_TOP_P=1.0; FAM_TOP_K=0
        ;;
    *)
        echo "WARN: unknown MODEL_FAMILY=$MODEL_FAMILY, defaulting to gemma4 sampling"
        FAM_TEMP=1.0; FAM_TOP_P=0.95; FAM_TOP_K=64
        ;;
esac

# --- Env overrides (individual knobs override family preset) ---
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

LLAMA="${LLAMA_BIN:-/opt/llama.cpp/build/bin}/llama-server"
PORT="${PORT:-8099}"
SEED="${SEED:-42}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
CONDA_ENV="${CONDA_ENV:-lightseek}"
TOKENIZER_PATH="${TOKENIZER_PATH:-google/gemma-4-26B-A4B-it}"
TEMP="${TEMP:-$FAM_TEMP}"
TOP_P="${TOP_P:-$FAM_TOP_P}"
TOP_K="${TOP_K:-$FAM_TOP_K}"
CTX="${CTX:-32768}"
REASONING_BUDGET="${REASONING_BUDGET:-16384}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-24576}"
# --dry-multiplier 0.5 is now a FIXED parameter (confirmed working on Q53:
# broke degenerate "re-re-re" token loop, model committed cleanly).
# DRY (Don't Repeat Yourself) sampler penalizes exact n-gram repeats.
DRY_MULT="${DRY_MULTIPLIER:-0.5}"

mkdir -p eval_results/server_logs eval_results/gpqa_full

LOG="eval_results/server_logs/eval_gpqa_${NAME}_server.log"
RUNLOG="eval_results/server_logs/eval_gpqa_${NAME}_run.log"
OUTPUT_DIR="eval_results/gpqa_full/${NAME}"
CACHE_DIR="eval_results/gpqa_cache"
CACHE_DB="${CACHE_DIR}/${NAME}"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

# Activate conda env. Try /root/anaconda3 first (solidpc), fallback to /opt/conda (pod docker).
if [[ -f /root/anaconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /root/anaconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV" 2>/dev/null || echo "  warning: conda env $CONDA_ENV not found, using base"
elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV" 2>/dev/null || echo "  using base conda (pod default)"
fi

# Verify lm_eval is callable
if ! command -v lm_eval >/dev/null 2>&1; then
    echo "ERROR: lm_eval not found on PATH after conda activate"
    exit 1
fi

LM_EVAL_VER=$(pip show lm-eval 2>/dev/null | grep Version | awk '{print $2}')
TRANSFORMERS_VER=$(pip show transformers 2>/dev/null | grep Version | awk '{print $2}')

echo "===== eval_gpqa_v3 (lm-eval direct): $NAME ====="
echo "  GGUF:             $GGUF"
echo "  MODEL_FAMILY:     $MODEL_FAMILY"
echo "  TEMP/TOP_P/TOP_K: $TEMP / $TOP_P / $TOP_K"
echo "  CTX / REASON_BDG: $CTX / $REASONING_BUDGET"
echo "  MAX_GEN_TOKS:     $MAX_GEN_TOKS"
echo "  PORT:             $PORT"
echo "  CUDA device:      $GPU_ID"
echo "  SEED:             $SEED"
echo "  SAMPLES:          ${SAMPLES:-(all 198)}"
echo "  DRY_MULTIPLIER:   ${DRY_MULT:-(disabled)}"
echo "  conda env:        $CONDA_ENV"
echo "  lm_eval ver:      $LM_EVAL_VER"
echo "  transformers:     $TRANSFORMERS_VER"
echo "  tokenizer:        $TOKENIZER_PATH"
echo "  log:              $LOG"
echo "  run log:          $RUNLOG"
echo "  output dir:       $OUTPUT_DIR"
echo "  cache db:         $CACHE_DB"
echo "  start:            $(date)"
echo

# Build optional --dry-multiplier flag
DRY_FLAG=""
if [[ -n "$DRY_MULT" ]]; then
    DRY_FLAG="--dry-multiplier $DRY_MULT"
fi

SPID=""
cleanup() {
    if [[ -n "$SPID" ]] && kill -0 $SPID 2>/dev/null; then
        kill $SPID 2>/dev/null || true
        wait $SPID 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Retry loop: if lm_eval crashes (llama.cpp PEG parse bug causes 500 → lm_eval dies
# after tenacity retries), we clean up and restart. Cache-based resume skips completed Qs.
MAX_RETRIES="${MAX_RETRIES:-6}"
RETRY_DELAY="${RETRY_DELAY:-15}"
EVAL_STATUS=1

for attempt in $(seq 1 $MAX_RETRIES); do
    echo
    echo "===== attempt $attempt / $MAX_RETRIES ====="

    # Cleanup any stale server from previous attempt or external process
    cleanup
    # Also kill any orphaned llama-server bound to our port (belt-and-suspenders)
    if pgrep -f "llama-server.*--port $PORT" >/dev/null 2>&1; then
        pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
        sleep 3
    fi
    # Wait for port to fully drain (TIME_WAIT etc)
    for i in $(seq 1 20); do
        if ! curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    # llama-server with current sampling config
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$LLAMA" \
        -m "$GGUF" \
        --port "$PORT" \
        -c "$CTX" \
        -t 12 \
        -ngl 99 \
        --no-warmup \
        --parallel 1 \
        --jinja \
        --reasoning-format deepseek \
        --reasoning-budget "$REASONING_BUDGET" \
        --temp "$TEMP" \
        --top-p "$TOP_P" \
        --top-k "$TOP_K" \
        --seed "$SEED" \
        $DRY_FLAG \
        >>"$LOG" 2>&1 &
    SPID=$!
    disown $SPID 2>/dev/null || true

    echo -n "  waiting for llama-server"
    READY=0
    for i in $(seq 1 240); do
        if curl -fsS "http://localhost:$PORT/health" 2>/dev/null | grep -q "ok"; then
            echo " — ready (PID $SPID)"
            READY=1
            break
        fi
        if ! kill -0 $SPID 2>/dev/null; then
            echo
            echo "  WARN: llama-server died during startup (attempt $attempt)"
            tail -10 "$LOG"
            break
        fi
        echo -n "."
        sleep 1
    done

    if [[ $READY -eq 0 ]]; then
        echo "  server never became healthy, retrying in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
        continue
    fi

    # lm_eval invocation — every param locked, no surprises
    echo
    echo "  launching lm_eval (attempt $attempt)..."
    set +e
    GEN_KWARGS="temperature=${TEMP},top_p=${TOP_P},max_gen_toks=${MAX_GEN_TOKS}"
    MODEL_ARGS="model=${NAME},base_url=http://localhost:${PORT}/v1/chat/completions,num_concurrent=1,tokenizer_backend=huggingface,tokenizer=${TOKENIZER_PATH},max_gen_toks=${MAX_GEN_TOKS}"
    if [[ -n "${SAMPLES:-}" ]]; then
        lm_eval \
            --model local-chat-completions \
            --model_args "$MODEL_ARGS" \
            --tasks gpqa_diamond_cot_zeroshot \
            --apply_chat_template \
            --batch_size 1 \
            --gen_kwargs "$GEN_KWARGS" \
            --log_samples \
            --use_cache "$CACHE_DB" \
            --samples "$SAMPLES" \
            --output_path "$OUTPUT_DIR" 2>&1 | tee -a "$RUNLOG"
        EVAL_STATUS=${PIPESTATUS[0]}
    else
        lm_eval \
            --model local-chat-completions \
            --model_args "$MODEL_ARGS" \
            --tasks gpqa_diamond_cot_zeroshot \
            --apply_chat_template \
            --batch_size 1 \
            --gen_kwargs "$GEN_KWARGS" \
            --log_samples \
            --use_cache "$CACHE_DB" \
            --output_path "$OUTPUT_DIR" 2>&1 | tee -a "$RUNLOG"
        EVAL_STATUS=${PIPESTATUS[0]}
    fi
    set -e

    if [[ $EVAL_STATUS -eq 0 ]]; then
        echo
        echo "===== lm_eval succeeded on attempt $attempt ====="
        break
    fi

    echo
    echo "===== lm_eval failed on attempt $attempt (exit $EVAL_STATUS) ====="
    echo "  likely llama.cpp PEG parse bug → server 500 → tenacity exhausted"
    echo "  tearing down server and retrying from cache in ${RETRY_DELAY}s..."
    sleep $RETRY_DELAY
done

cleanup
trap - EXIT INT TERM

if [[ $EVAL_STATUS -ne 0 ]]; then
    echo "ERROR: all $MAX_RETRIES attempts failed for $NAME"
    exit 1
fi

echo
echo "===== $NAME done — $(date) ====="
echo "  output:  $OUTPUT_DIR"
