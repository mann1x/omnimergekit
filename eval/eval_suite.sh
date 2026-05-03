#!/bin/bash
# Unified evaluation suite for Qwen3.5-27B models via llama.cpp + lm-eval-harness.
#
# Usage:
#   eval_suite.sh <model_name> <gguf_path> [--tokenizer <path>] [--port <port>]
#                 [--benchmarks <list>] [--gpqa-mode] [--skip-server]
#
# Examples:
#   # Full suite, auto-start server:
#   eval_suite.sh Omnimerge-v4 /workspace/gguf/Omnimerge-Q6_K.gguf --tokenizer /workspace/merged_omnimerge
#
#   # Only fast benchmarks:
#   eval_suite.sh Claude-distill /path/to/claude.gguf --benchmarks humaneval,mbpp
#
#   # GPQA Diamond with reasoning mode (separate server config):
#   eval_suite.sh Omnimerge-v4 /path/to/omni.gguf --gpqa-mode --tokenizer /workspace/merged_omnimerge
#
#   # Use existing server on port 8099:
#   eval_suite.sh Omnimerge-v4 unused --skip-server --tokenizer /workspace/merged_omnimerge
#
# Benchmarks available:
#   humaneval     - HumanEval pass@1 (164q, ~10 min, generation, code completion)
#   mbpp          - MBPP pass@1 (500q, ~25 min, generation, code)
#   ifeval        - IFEval (541q, ~30-60 min, generation, instruction following)
#   gsm8k         - GSM8K CoT zeroshot (1319q full / 200 with --limit, math reasoning)
#   gpqa          - GPQA Diamond CoT zeroshot (198q, ~3h, generation, reasoning)
#
# NOTE: MCQ benchmarks (ARC-Challenge, HellaSwag, WinoGrande, MMLU) are NOT supported
# via llama.cpp API. They require loglikelihood scoring which needs a native model backend
# (HF transformers or vLLM), not an OpenAI-compatible API. lm-eval's local-completions
# and local-chat-completions models raise NotImplementedError on loglikelihood requests.
#
# Server modes:
#   Default (code/MCQ):  plain server, no --jinja, no --reasoning-format
#   GPQA mode:           --jinja --reasoning-format deepseek --reasoning-budget 16384
#   The script handles switching automatically when gpqa is in the benchmark list.
#
# Requirements:
#   - llama.cpp server at $LLAMA_BIN or /opt/llama.cpp/build/bin
#   - lm_eval (pip install lm-eval)
#   - HF_ALLOW_CODE_EVAL=1 set for humaneval/mbpp
#   - For GPQA: --reasoning-format deepseek requires llama.cpp with jinja support

set -uo pipefail

# --- Defaults ---
MODEL_NAME="${1:?Usage: eval_suite.sh <model_name> <gguf_path> [options]}"
GGUF_PATH="${2:?Usage: eval_suite.sh <model_name> <gguf_path> [options]}"
shift 2

PORT=8099
TOKENIZER=""
BENCHMARKS="humaneval,mbpp,gsm8k"
GPQA_MODE=0
SKIP_SERVER=0
RESULTS_DIR="${RESULTS_DIR:-eval_results}"
CACHE_DIR="${RESULTS_DIR}/cache"
LLAMA_BIN="${LLAMA_BIN:-/opt/llama.cpp/build/bin}"
MAX_RETRIES=3

# Find lm_eval binary: LM_EVAL env > conda lightseek > lightseek venv > PATH
if [[ -n "${LM_EVAL:-}" ]] && [[ -x "$LM_EVAL" ]]; then
    :
elif [[ -x "/shared/dev/lightseek/.venv/bin/lm_eval" ]]; then
    LM_EVAL="/shared/dev/lightseek/.venv/bin/lm_eval"
elif command -v lm_eval &>/dev/null; then
    LM_EVAL="$(command -v lm_eval)"
elif [[ -x "/opt/conda/bin/lm_eval" ]]; then
    LM_EVAL="/opt/conda/bin/lm_eval"
else
    echo "ERROR: lm_eval not found. Set LM_EVAL=/path/to/lm_eval"
    exit 1
fi
echo "  lm_eval: $LM_EVAL"

# Ensure lm-eval optional dependencies are installed (IFEval needs these)
LM_EVAL_PIP="$(dirname "$LM_EVAL")/pip"
if [[ ! -x "$LM_EVAL_PIP" ]]; then
    LM_EVAL_PIP="$(dirname "$(dirname "$LM_EVAL")")/bin/pip"
fi
if [[ -x "$LM_EVAL_PIP" ]]; then
    for pkg in langdetect immutabledict nltk; do
        "$LM_EVAL_PIP" show "$pkg" &>/dev/null || {
            echo "  installing missing dep: $pkg"
            "$LM_EVAL_PIP" install -q "$pkg" 2>/dev/null
        }
    done
fi

GSM8K_LIMIT=""  # empty = full 1319, or set e.g. 200 for quick run

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --tokenizer)    TOKENIZER="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --benchmarks)   BENCHMARKS="$2"; shift 2 ;;
        --gsm8k-limit)  GSM8K_LIMIT="$2"; shift 2 ;;
        --gpqa-mode)    GPQA_MODE=1; shift ;;
        --skip-server)  SKIP_SERVER=1; shift ;;
        --results-dir)  RESULTS_DIR="$2"; CACHE_DIR="$RESULTS_DIR/cache"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Auto-detect tokenizer from GGUF directory if not specified
if [[ -z "$TOKENIZER" ]]; then
    GGUF_DIR=$(dirname "$GGUF_PATH")
    if [[ -f "$GGUF_DIR/tokenizer.json" ]]; then
        TOKENIZER="$GGUF_DIR"
    else
        echo "ERROR: --tokenizer required (no tokenizer.json found next to GGUF)"
        exit 1
    fi
fi

export HF_ALLOW_CODE_EVAL=1
mkdir -p "$RESULTS_DIR/server_logs" "$CACHE_DIR"

LOG="$RESULTS_DIR/server_logs/eval_suite_${MODEL_NAME}.log"
exec > >(tee -a "$LOG") 2>&1

echo "===== $(date) EVAL SUITE: $MODEL_NAME ====="
echo "  gguf:       $GGUF_PATH"
echo "  tokenizer:  $TOKENIZER"
echo "  benchmarks: $BENCHMARKS"
echo "  port:       $PORT"
echo "  gpqa_mode:  $GPQA_MODE"
echo "  results:    $RESULTS_DIR"
echo

# --- Helper: start llama-server ---
start_server() {
    local mode="$1"  # "plain" or "gpqa"

    # Kill any existing server on this port
    pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
    sleep 2

    local SERVER_ARGS="-m $GGUF_PATH --port $PORT -c 32768 -t 12 -ngl 99 --no-warmup --parallel 1"
    SERVER_ARGS+=" --temp 0.6 --top-p 0.95 --top-k 20 --seed 42 --dry-multiplier 0.5"

    if [[ "$mode" == "gpqa" ]]; then
        SERVER_ARGS+=" --jinja --reasoning-format deepseek --reasoning-budget 16384"
        echo "  server mode: GPQA (reasoning-format deepseek, budget 16384)"
    else
        echo "  server mode: plain (no reasoning flags)"
    fi

    setsid "$LLAMA_BIN/llama-server" $SERVER_ARGS \
        </dev/null >"$RESULTS_DIR/server_logs/server_${MODEL_NAME}_${mode}.log" 2>&1 &
    disown

    # Wait for health
    for i in $(seq 1 60); do
        local health
        health=$(curl -s -m 3 "http://localhost:$PORT/health" 2>/dev/null)
        if [[ "$health" == *'"status":"ok"'* ]]; then
            echo "  server ready (attempt $i)"
            return 0
        fi
        sleep 3
    done
    echo "ERROR: server failed to start"
    return 1
}

# --- Helper: run a benchmark ---
run_benchmark() {
    local task="$1"
    local model_type="$2"   # local-completions or local-chat-completions
    local base_url="$3"
    local max_gen="$4"
    local extra_args="$5"

    local task_cache="$CACHE_DIR/${MODEL_NAME}_${task}"
    local task_output="$RESULTS_DIR/$task/$MODEL_NAME"

    echo
    echo "=== $(date) BENCHMARK: $task ==="
    echo "  model_type: $model_type"
    echo "  max_gen:    $max_gen"

    local attempt=0
    while [[ $attempt -lt $MAX_RETRIES ]]; do
        attempt=$((attempt + 1))
        echo "  attempt $attempt/$MAX_RETRIES"

        "$LM_EVAL" \
            --model "$model_type" \
            --model_args "model=$MODEL_NAME,base_url=$base_url,num_concurrent=1,tokenizer_backend=huggingface,tokenizer=$TOKENIZER,max_length=32768,max_gen_toks=$max_gen" \
            --tasks "$task" \
            --batch_size 1 \
            --log_samples \
            --confirm_run_unsafe_code \
            --use_cache "$task_cache" \
            --output_path "$task_output" \
            $extra_args \
            2>&1

        local rc=$?
        if [[ $rc -eq 0 ]]; then
            echo "  $task completed successfully"

            # Extract and display score
            local result_json
            result_json=$(ls "$task_output/$MODEL_NAME/results_"*.json 2>/dev/null | tail -1)
            if [[ -n "$result_json" ]]; then
                python3 -c "
import json
d = json.load(open('$result_json'))
for task, r in d['results'].items():
    for k, v in r.items():
        if 'stderr' not in k:
            print(f'  [SCORE] {task}/{k}: {v}')
"
            fi
            return 0
        fi

        echo "  WARN: $task failed (rc=$rc), retrying in 15s..."
        sleep 15
    done

    echo "  ERROR: $task failed after $MAX_RETRIES attempts"
    return 1
}

# --- Sanity check: verify samples after code tasks ---
sanity_check() {
    local task="$1"
    local task_output="$RESULTS_DIR/$task/$MODEL_NAME"

    local samples
    samples=$(ls "$task_output/$MODEL_NAME/samples_"*.jsonl 2>/dev/null | tail -1)
    if [[ -z "$samples" ]]; then
        echo "  [sanity] no samples file found"
        return
    fi

    python3 -c "
import json, re

total = passed = empty = fence = degen = 0
with open('$samples') as f:
    for line in f:
        s = json.loads(line)
        total += 1
        resps = s.get('resps', [['']])
        raw = resps[0][0] if resps and resps[0] else ''
        if len(raw) < 10: empty += 1
        if '\`\`\`' in raw: fence += 1
        if len(raw) > 200:
            tail = raw[-200:]
            for cl in range(15, 40):
                if tail.count(tail[-cl:]) >= 4:
                    degen += 1
                    break

print(f'  [sanity] {task}: total={total} empty={empty} fence={fence} degen={degen}')
if empty > 0: print(f'  [WARN] {empty} empty responses!')
if degen > 0: print(f'  [WARN] {degen} degenerate loops!')
" 2>/dev/null
}

# --- Split benchmarks into groups by server mode ---
IFS=',' read -ra BENCH_LIST <<< "$BENCHMARKS"

PLAIN_TASKS=()
GPQA_TASKS=()
for b in "${BENCH_LIST[@]}"; do
    b=$(echo "$b" | xargs)  # trim whitespace
    case $b in
        gpqa|gpqa_diamond|gpqa_diamond_cot_zeroshot)
            GPQA_TASKS+=("gpqa_diamond_cot_zeroshot")
            ;;
        *)
            PLAIN_TASKS+=("$b")
            ;;
    esac
done

# --- Run plain-mode benchmarks (code + MCQ) ---
if [[ ${#PLAIN_TASKS[@]} -gt 0 ]]; then
    echo
    echo "===== PLAIN-MODE BENCHMARKS: ${PLAIN_TASKS[*]} ====="

    if [[ $SKIP_SERVER -eq 0 ]]; then
        start_server "plain" || exit 1
    fi

    BASE_URL="http://localhost:$PORT/v1/completions"

    for task in "${PLAIN_TASKS[@]}"; do
        case $task in
            humaneval)
                # gen_kwargs until adds ``` as a stop sequence — merged models sometimes
                # emit a trailing ``` that causes SyntaxError in exec() scoring.
                run_benchmark "humaneval" "local-completions" "$BASE_URL" 2048 '--gen_kwargs "until=[\"\\n\\nclass \",\"\\n\\ndef \",\"\\n\\n#\",\"\\n\\nif \",\"\\n\\nprint\",\"\\`\\`\\`\"]"'
                sanity_check "humaneval"
                ;;
            mbpp)
                run_benchmark "mbpp" "local-completions" "$BASE_URL" 2048 '--gen_kwargs "until=[\"\\n\\nclass \",\"\\n\\ndef \",\"\\n\\n#\",\"\\n\\nif \",\"\\n\\nprint\",\"\\`\\`\\`\"]"'
                sanity_check "mbpp"
                ;;
            ifeval)
                run_benchmark "ifeval" "local-completions" "$BASE_URL" 4096 ""
                ;;
            gsm8k|gsm8k_cot_zeroshot)
                local gsm8k_extra=""
                if [[ -n "$GSM8K_LIMIT" ]]; then
                    gsm8k_extra="--limit $GSM8K_LIMIT"
                    echo "  GSM8K limited to $GSM8K_LIMIT questions"
                fi
                run_benchmark "gsm8k_cot_zeroshot" "local-completions" "$BASE_URL" 4096 "$gsm8k_extra"
                ;;
            *)
                echo "WARNING: unknown benchmark '$task' — supported: humaneval, mbpp, ifeval, gsm8k, gpqa"
                ;;
        esac
    done
fi

# --- Run GPQA-mode benchmarks (reasoning) ---
if [[ ${#GPQA_TASKS[@]} -gt 0 ]]; then
    echo
    echo "===== GPQA-MODE BENCHMARKS: ${GPQA_TASKS[*]} ====="

    if [[ $SKIP_SERVER -eq 0 ]]; then
        start_server "gpqa" || exit 1
    fi

    # GPQA uses /v1/completions to avoid PEG parser crash, but server has
    # --reasoning-format deepseek --reasoning-budget 16384 for budget enforcement.
    BASE_URL="http://localhost:$PORT/v1/completions"

    for task in "${GPQA_TASKS[@]}"; do
        run_benchmark "$task" "local-completions" "$BASE_URL" 16384 "--apply_chat_template"
    done
fi

# --- Summary ---
echo
echo "===== $(date) EVAL SUITE COMPLETE: $MODEL_NAME ====="
echo
echo "Results:"

python3 -c "
import json, glob, os

results_dir = '$RESULTS_DIR'
model = '$MODEL_NAME'

# Find all result JSONs
patterns = [
    f'{results_dir}/*/\{model}/\{model}/results_*.json',
    f'{results_dir}/*/{model}/results_*.json',
]
files = []
for p in patterns:
    files.extend(glob.glob(p))

if not files:
    print('  No results found')
    exit()

scores = {}
for f in sorted(set(files)):
    d = json.load(open(f))
    for task, r in d.get('results', {}).items():
        for k, v in r.items():
            if 'stderr' not in k and 'alias' not in k:
                scores[f'{task}/{k}'] = v

print(f'  {\"Benchmark\":<50s} {\"Score\":>8s}')
print(f'  {\"-\"*50} {\"-\"*8}')
for k, v in sorted(scores.items()):
    if isinstance(v, float):
        print(f'  {k:<50s} {v:>8.4f}')
    else:
        print(f'  {k:<50s} {str(v):>8s}')
" 2>/dev/null

echo
echo "===== DONE ====="
