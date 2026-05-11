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
# Server profiles:
#   Each benchmark family runs the llama-server with a distinct, named profile.
#   Profiles are bash variables defined in the Defaults section and can be
#   overridden via CLI or env. The script restarts the server between groups
#   so each eval gets the right sampler / template / reasoning settings, and
#   shuts down the server cleanly at the end.
#
#     plain  — raw /v1/completions, greedy, no chat template.
#              Tasks: humaneval, mbpp, ifeval, gsm8k, *_main, *_subset.
#     chat   — /v1/chat/completions for IT models, --jinja --reasoning off.
#              Tasks: *_chat, *_instruct.
#     gpqa   — /v1/completions with --jinja --reasoning-format deepseek
#              --reasoning-budget 16384 (mandatory per CLAUDE.md).
#              Tasks: gpqa, gpqa_diamond_*, gpqa_*_smoke*.
#     gsm8k  — same as plain by default; reserved for future divergence.
#
#   Override individual profiles:
#     --plain-server-args "..."   --chat-server-args "..."
#     --gpqa-server-args "..."    --gsm8k-server-args "..."
#   Or via env vars: PROFILE_PLAIN / PROFILE_CHAT / PROFILE_GPQA / PROFILE_GSM8K
#
#   Override common base flags (applied to every profile):
#     --ctx N    --ngl N    --threads N    --parallel N
#     --server-base-args "<extra base flags>"
#     --server-args "<extra appended after profile>"
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
# Default include path resolves to <repo>/eval/tasks/ (the dir holding our
# curated subset YAMLs and _subset_filter.py). Override with --include-path
# to point lm-eval at additional task definitions.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INCLUDE_PATH="${INCLUDE_PATH:-$SCRIPT_DIR/tasks}"

# ── llama-server profiles ────────────────────────────────────────────────
# Each profile is a complete set of sampler / template / reasoning flags
# tailored to a benchmark family. They are applied on top of SERVER_BASE_ARGS
# (which carries -m / --port / -c / -t / -ngl / --no-warmup / --parallel).
# Override any profile from env or CLI: env vars take precedence over
# defaults; --plain-server-args / --chat-server-args / --gpqa-server-args /
# --gsm8k-server-args / --server-args set them at invocation time.
#
# PROFILE_PLAIN  — raw /v1/completions, code-completion or MCQ on base/non-IT
#                  models. Greedy, no chat template.
# PROFILE_CHAT   — /v1/chat/completions for IT models on HE/MBPP. jinja chat
#                  template + `--reasoning off` to disable thinking (Gemma 4
#                  IT and similar would otherwise route output to
#                  reasoning_content; bug-125).
# PROFILE_GPQA   — /v1/completions but with full reasoning enabled
#                  (deepseek format) and a token budget. Per CLAUDE.md the
#                  budget is mandatory: without it Gemma 4 enters
#                  overthinking loops.
# PROFILE_GSM8K  — math reasoning via plain prompt; greedy.
PROFILE_PLAIN="${PROFILE_PLAIN:---temp 0 --top-p 1 --top-k 0 --seed 42}"
PROFILE_CHAT="${PROFILE_CHAT:---temp 0 --top-p 1 --top-k 0 --seed 42 --jinja --reasoning off}"
PROFILE_GPQA="${PROFILE_GPQA:---temp 0.6 --top-p 0.95 --top-k 20 --seed 42 --dry-multiplier 0.5 --jinja --reasoning-format deepseek --reasoning-budget 16384}"
PROFILE_GSM8K="${PROFILE_GSM8K:---temp 0 --top-p 1 --top-k 0 --seed 42}"

# Server-base flags — applied to every profile. Override with --ctx / --ngl /
# --threads or --server-base-args.
CTX="${CTX:-32768}"
NGL="${NGL:-99}"
THREADS="${THREADS:-12}"
PARALLEL="${PARALLEL:-1}"
SERVER_BASE_ARGS=""              # extra base-level flags (always applied)
SERVER_ARGS_EXTRA=""             # extra flags appended after the profile

# Find lm_eval binary. Preference order:
#   1. LM_EVAL env override
#   2. omnimergekit conda env (canonical env for this repo)
#   3. lightseek .venv (local fallback on solidPC)
#   4. anything on PATH
#   5. /opt/conda (pod default)
if [[ -n "${LM_EVAL:-}" ]] && [[ -x "$LM_EVAL" ]]; then
    :
elif [[ -x "/root/anaconda3/envs/omnimergekit/bin/lm_eval" ]]; then
    LM_EVAL="/root/anaconda3/envs/omnimergekit/bin/lm_eval"
elif [[ -x "/opt/conda/envs/omnimergekit/bin/lm_eval" ]]; then
    LM_EVAL="/opt/conda/envs/omnimergekit/bin/lm_eval"
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
        --include-path) INCLUDE_PATH="$2"; shift 2 ;;
        # Server config
        --ctx)             CTX="$2"; shift 2 ;;
        --ngl)             NGL="$2"; shift 2 ;;
        --threads)         THREADS="$2"; shift 2 ;;
        --parallel)        PARALLEL="$2"; shift 2 ;;
        --server-base-args) SERVER_BASE_ARGS="$2"; shift 2 ;;
        --server-args)     SERVER_ARGS_EXTRA="$2"; shift 2 ;;
        # Per-profile complete overrides (replace the whole profile string)
        --plain-server-args) PROFILE_PLAIN="$2"; shift 2 ;;
        --chat-server-args)  PROFILE_CHAT="$2"; shift 2 ;;
        --gpqa-server-args)  PROFILE_GPQA="$2"; shift 2 ;;
        --gsm8k-server-args) PROFILE_GSM8K="$2"; shift 2 ;;
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
echo "  include:    $INCLUDE_PATH"
echo "  server-base: -c $CTX -t $THREADS -ngl $NGL --parallel $PARALLEL ${SERVER_BASE_ARGS:+($SERVER_BASE_ARGS)}"
[[ -n "$SERVER_ARGS_EXTRA" ]] && echo "  server-extra: $SERVER_ARGS_EXTRA"
echo

# --- Helper: start llama-server ---
stop_server() {
    # Kill any llama-server we own — match by both port AND GGUF path so we
    # only target the one we (or a prior eval_suite invocation) launched, not
    # an unrelated server. SIGTERM first, escalate to KILL if it lingers.
    local pids
    pids=$(pgrep -f "llama-server.*--port $PORT" 2>/dev/null || true)
    [[ -z "$pids" ]] && return 0
    echo "  stopping llama-server PIDs: $pids"
    kill $pids 2>/dev/null || true
    for _ in 1 2 3 4 5; do
        sleep 1
        pgrep -f "llama-server.*--port $PORT" >/dev/null 2>&1 || return 0
    done
    echo "  escalating to SIGKILL"
    pkill -9 -f "llama-server.*--port $PORT" 2>/dev/null || true
    sleep 1
}

# Wait for VRAM to drop below threshold (in MiB) — protects against the
# zombie-server pattern where the OS hasn't reclaimed CUDA buffers yet.
wait_for_vram_clear() {
    local threshold="${1:-1024}"
    local max_wait="${2:-30}"
    local i
    for ((i=0; i<max_wait; i++)); do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        [[ -z "$used" ]] && return 0  # no GPU? skip the check
        if (( used < threshold )); then
            return 0
        fi
        sleep 1
    done
    echo "  WARN: VRAM still ${used} MiB after ${max_wait}s — proceeding anyway"
}

start_server() {
    local profile="$1"  # "plain" | "chat" | "gpqa" | "gsm8k"

    stop_server
    wait_for_vram_clear 1024 30

    local base_args="-m $GGUF_PATH --port $PORT -c $CTX -t $THREADS -ngl $NGL --no-warmup --parallel $PARALLEL"
    [[ -n "$SERVER_BASE_ARGS" ]] && base_args+=" $SERVER_BASE_ARGS"

    local profile_args=""
    case "$profile" in
        plain)  profile_args="$PROFILE_PLAIN";  echo "  server profile: plain  ($profile_args)" ;;
        chat)   profile_args="$PROFILE_CHAT";   echo "  server profile: chat   ($profile_args)" ;;
        gpqa)   profile_args="$PROFILE_GPQA";   echo "  server profile: gpqa   ($profile_args)" ;;
        gsm8k)  profile_args="$PROFILE_GSM8K";  echo "  server profile: gsm8k  ($profile_args)" ;;
        *)      echo "ERROR: unknown profile '$profile'"; return 1 ;;
    esac

    local SERVER_ARGS="$base_args $profile_args"
    [[ -n "$SERVER_ARGS_EXTRA" ]] && SERVER_ARGS+=" $SERVER_ARGS_EXTRA"

    # Log the full invocation: shows in eval_suite log AND is written as the
    # first line of the server log so the binding between server output and
    # its launch arguments is unambiguous when reviewing later.
    local server_log="$RESULTS_DIR/server_logs/server_${MODEL_NAME}_${profile}.log"
    local cmdline="$LLAMA_BIN/llama-server $SERVER_ARGS"
    echo "  llama-server cmd: $cmdline"
    {
        echo "###############################################################"
        echo "# eval_suite.sh start_server profile='$profile' at $(date -Iseconds)"
        echo "# cmdline:"
        echo "#   $cmdline"
        echo "# pid will follow on its first stdout line"
        echo "###############################################################"
    } > "$server_log"

    setsid "$LLAMA_BIN/llama-server" $SERVER_ARGS \
        </dev/null >>"$server_log" 2>&1 &
    local server_pid=$!
    disown
    echo "  llama-server pid: $server_pid (log: $server_log)"

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

        local include_args=""
        if [[ -n "$INCLUDE_PATH" && -d "$INCLUDE_PATH" ]]; then
            include_args="--include_path $INCLUDE_PATH"
        fi
        "$LM_EVAL" \
            --model "$model_type" \
            --model_args "model=$MODEL_NAME,base_url=$base_url,num_concurrent=${NUM_CONCURRENT:-1},tokenizer_backend=huggingface,tokenizer=$TOKENIZER,max_length=32768,max_gen_toks=$max_gen" \
            --tasks "$task" \
            --batch_size 1 \
            --log_samples \
            --confirm_run_unsafe_code \
            --use_cache "$task_cache" \
            --output_path "$task_output" \
            $include_args \
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
CHAT_TASKS=()
GPQA_TASKS=()
for b in "${BENCH_LIST[@]}"; do
    b=$(echo "$b" | xargs)  # trim whitespace
    case $b in
        gpqa|gpqa_diamond|gpqa_diamond_cot_zeroshot)
            GPQA_TASKS+=("gpqa_diamond_cot_zeroshot")
            ;;
        gpqa_*|*_gpqa_*)
            # Custom GPQA-derived tasks (smoke subsets, etc.) — route to GPQA-mode server.
            GPQA_TASKS+=("$b")
            ;;
        *_chat|*_instruct)
            # Chat-completions code tasks — IT models on HE/MBPP must go through
            # /v1/chat/completions with apply_chat_template, not raw /v1/completions.
            CHAT_TASKS+=("$b")
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
            humaneval|humaneval_*)
                # The YAML's `until` list (\nclass, \ndef, \n#, \nif, \nprint) covers normal
                # completions. Trailing ``` from merged-model degenerates is post-hoc
                # handled by eval/rescore_humaneval_strip_fences.py.
                # NOTE: lm-eval 0.5+ rejects --gen_kwargs values containing top-level commas
                # (key_val_to_dict splits on `,`), so we no longer override the until list
                # at the CLI — the YAML default suffices.
                run_benchmark "$task" "local-completions" "$BASE_URL" 2048 ""
                sanity_check "$task"
                ;;
            mbpp|mbpp_*)
                # See note above — rely on YAML default `until: [DONE]`.
                run_benchmark "$task" "local-completions" "$BASE_URL" 2048 ""
                sanity_check "$task"
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

# --- Run chat-mode benchmarks (HE/MBPP for IT models, *_chat / *_instruct) ---
if [[ ${#CHAT_TASKS[@]} -gt 0 ]]; then
    echo
    echo "===== CHAT-MODE BENCHMARKS: ${CHAT_TASKS[*]} ====="

    if [[ $SKIP_SERVER -eq 0 ]]; then
        start_server "chat" || exit 1
    fi

    BASE_URL="http://localhost:$PORT/v1/chat/completions"

    for task in "${CHAT_TASKS[@]}"; do
        case $task in
            humaneval_*|*humaneval*_chat|*humaneval*_instruct)
                run_benchmark "$task" "local-chat-completions" "$BASE_URL" 1024 "--apply_chat_template"
                sanity_check "$task"
                ;;
            mbpp_*|*mbpp*_chat|*mbpp*_instruct)
                run_benchmark "$task" "local-chat-completions" "$BASE_URL" 512 "--apply_chat_template"
                sanity_check "$task"
                ;;
            *)
                run_benchmark "$task" "local-chat-completions" "$BASE_URL" 2048 "--apply_chat_template"
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
# Final cleanup: stop any llama-server we launched. SKIP_SERVER means the
# caller is managing the server lifecycle externally — leave it alone.
if [[ $SKIP_SERVER -eq 0 ]]; then
    echo "===== Cleanup ====="
    stop_server
fi

echo
echo "===== DONE ====="
