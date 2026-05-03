#!/bin/bash
# Download the 4 Qwen3.5-27B frankenmerge candidates and run HumanEval on each.
# Phase 1 of the Omnimerge experiment — establishes individual baselines before merging.
#
# For each model:
#   1) Download HF repo (lazy: skip if already there)
#   2) Convert to F16 GGUF
#   3) Quantize to Q6_K
#   4) Delete F16 GGUF (save disk)
#   5) Run HumanEval via llama-server + lm-eval local-chat-completions
#   6) Delete Q6_K GGUF (keep HF for later merge)
#   7) Append result to /workspace/humaneval_sources.log
#
# After all 4: print combined table.
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN required}"

MODELS=(
    "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"
    "DavidAU/Qwen3.5-27B-Gemini3-Pro-High-Reasoning-Compact-Thinking"
    "ConicCat/Qwen3.5-27B-Writer-V2"
    "ValiantLabs/Qwen3.5-27B-Esper3.1"
)

WORKDIR=/workspace
HF_ROOT=$WORKDIR/hf_models
GGUF_ROOT=$WORKDIR/gguf
LOG=$WORKDIR/humaneval_sources.log
LLAMA_BIN=/opt/llama.cpp/build/bin

mkdir -p "$HF_ROOT" "$GGUF_ROOT"

echo "===== $(date) QWEN3.5-27B SOURCE EVAL START =====" | tee -a "$LOG"
echo "Models: ${MODELS[*]}" | tee -a "$LOG"

# --- Phase 1: parallel downloads ---
echo | tee -a "$LOG"
echo "=== Phase 1: parallel download of all 4 HF repos ===" | tee -a "$LOG"
for m in "${MODELS[@]}"; do
    name=$(basename "$m")
    if [[ -d "$HF_ROOT/$name" && -f "$HF_ROOT/$name/config.json" ]]; then
        echo "  $name: already downloaded, skipping" | tee -a "$LOG"
        continue
    fi
    echo "  $name: downloading in background..." | tee -a "$LOG"
    (
        python3 - <<PY >>$WORKDIR/download_${name}.log 2>&1
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="$m",
    local_dir="$HF_ROOT/$name",
    ignore_patterns=["*.bin", "*.pt", "consolidated.*", "*.gguf"],
)
print("downloaded", p)
PY
    ) &
done
wait
echo "  all downloads complete at $(date)" | tee -a "$LOG"

# --- Phase 2: per-model convert + HumanEval ---
for m in "${MODELS[@]}"; do
    name=$(basename "$m")
    hf_dir="$HF_ROOT/$name"
    f16="$GGUF_ROOT/${name}-F16.gguf"
    q6k="$GGUF_ROOT/${name}-Q6_K.gguf"

    echo | tee -a "$LOG"
    echo "=== $(date) EVAL: $name ===" | tee -a "$LOG"

    # Convert → F16
    if [[ ! -f "$f16" && ! -f "$q6k" ]]; then
        echo "  [1/3] convert to F16..." | tee -a "$LOG"
        python3 /opt/llama.cpp/convert_hf_to_gguf.py "$hf_dir" \
            --outfile "$f16" --outtype f16 2>&1 | tail -5 | tee -a "$LOG"
    fi

    # Quantize → Q6_K
    if [[ ! -f "$q6k" ]]; then
        echo "  [2/3] quantize to Q6_K..." | tee -a "$LOG"
        "$LLAMA_BIN/llama-quantize" "$f16" "$q6k" Q6_K 2>&1 | tail -5 | tee -a "$LOG"
    fi

    # Delete F16 to save disk
    [[ -f "$f16" ]] && rm -f "$f16" && echo "  removed F16" | tee -a "$LOG"
    ls -la "$q6k" | tee -a "$LOG"

    # Start llama-server for this model.
    # NOTE: no --jinja (we use /v1/completions, not /v1/chat/completions — no chat template).
    echo "  [3/3] HumanEval via llama-server + lm-eval (raw completions)..." | tee -a "$LOG"
    "$LLAMA_BIN/llama-server" \
        -m "$q6k" \
        --port 8099 -c 32768 -ngl 99 --no-warmup --parallel 1 \
        --seed 42 \
        > $WORKDIR/${name}_server.log 2>&1 &
    SPID=$!
    # Wait for health
    for i in $(seq 1 120); do
        if curl -fsS http://localhost:8099/health 2>/dev/null | grep -q ok; then
            echo "  server ready (PID $SPID)" | tee -a "$LOG"
            break
        fi
        if ! kill -0 $SPID 2>/dev/null; then
            echo "  server died, skipping" | tee -a "$LOG"
            break
        fi
        sleep 1
    done

    # Run HumanEval via RAW text-completion API (not chat).
    # Background: Qwen3.5 reasoning models in chat mode always wrap code in
    # ```python ... ``` markdown fences that make `prompt+generation` unparseable
    # by humaneval's scorer → pass@1 = 0 across the board. They also emit <think>
    # tokens unconditionally regardless of /no_think or enable_thinking=False.
    # The base text-completion endpoint bypasses the chat template entirely and
    # Qwen3.5 (built on Qwen2.5-Coder base) does clean fill-in-the-middle completion
    # on the raw humaneval prompt. This is also the canonical humaneval setup.
    #
    # Resumable: lm-eval's --use_cache creates a SQLite db of (model, request) →
    # response. If a prior partial run dies, subsequent runs skip already-computed
    # requests. Cache is keyed by the `model=` arg in model_args, so we use the
    # model name (not the file path) so retries match exactly.
    #
    # Retry loop: if llama-server dies (OOM, PEG parser, whatever), tear down and
    # restart. Cache means zero progress loss between retries.
    export HF_ALLOW_CODE_EVAL=1
    mkdir -p "$WORKDIR/humaneval_cache"
    CACHE_PREFIX="$WORKDIR/humaneval_cache/${name}"

    MAX_RETRIES=${MAX_RETRIES:-6}
    RETRY_DELAY=${RETRY_DELAY:-15}
    EVAL_OK=0
    for attempt in $(seq 1 $MAX_RETRIES); do
        echo "  [eval attempt $attempt/$MAX_RETRIES] lm_eval humaneval..." | tee -a "$LOG"
        set +e
        lm_eval \
            --model local-completions \
            --model_args "model=${name},base_url=http://localhost:8099/v1/completions,num_concurrent=1,tokenizer_backend=huggingface,tokenizer=$hf_dir,max_gen_toks=1024" \
            --tasks humaneval \
            --batch_size 1 \
            --confirm_run_unsafe_code \
            --use_cache "$CACHE_PREFIX" \
            --log_samples \
            --gen_kwargs "temperature=0.0,max_gen_toks=1024" \
            --output_path "$WORKDIR/humaneval_${name}" 2>&1 | tee -a "$LOG"
        RC=${PIPESTATUS[0]}
        set -e
        if [[ $RC -eq 0 ]]; then
            EVAL_OK=1
            break
        fi
        echo "  lm_eval exited $RC on attempt $attempt, retrying in ${RETRY_DELAY}s..." | tee -a "$LOG"
        # Teardown + restart server — cache survives, resume point is whatever made it into SQLite
        kill $SPID 2>/dev/null || true
        wait $SPID 2>/dev/null || true
        sleep $RETRY_DELAY
        # Restart server for retry
        "$LLAMA_BIN/llama-server" \
            -m "$q6k" \
            --port 8099 -c 32768 -ngl 99 --no-warmup --parallel 1 \
            --seed 42 \
            >> $WORKDIR/${name}_server.log 2>&1 &
        SPID=$!
        for i in $(seq 1 120); do
            curl -fsS http://localhost:8099/health 2>/dev/null | grep -q ok && break
            kill -0 $SPID 2>/dev/null || { echo "  retry server died" | tee -a "$LOG"; break; }
            sleep 1
        done
    done

    if [[ $EVAL_OK -ne 1 ]]; then
        echo "  HumanEval FAILED for $name after $MAX_RETRIES attempts" | tee -a "$LOG"
    fi

    # Sanity check: validate the samples jsonl contains real Python code (not
    # markdown, not empty) and a real pass@1 number.
    SAMP=$(ls "$WORKDIR/humaneval_${name}"/**/samples_humaneval_*.jsonl 2>/dev/null | head -1 || true)
    RES=$(ls "$WORKDIR/humaneval_${name}"/**/results_*.json 2>/dev/null | head -1 || true)
    if [[ -n "$SAMP" && -n "$RES" ]]; then
        python3 - <<PY | tee -a "$LOG"
import json
samp, res = "$SAMP", "$RES"
with open(res) as f:
    d = json.load(f)
r = list(d["results"].values())[0]
score = next((v for k,v in r.items() if "pass" in k.lower() and "stderr" not in k), None)
total = sum(1 for _ in open(samp))
bad = fence = empty = 0
import re
with open(samp) as f:
    for line in f:
        s = json.loads(line)
        gen = s.get("resps", [[""]])[0][0] if s.get("resps") else ""
        if not gen.strip(): empty += 1
        if "\`\`\`" in gen: fence += 1
        if len(gen.strip()) < 5: bad += 1
print(f"  [sanity] $name: pass@1={score}  samples={total}  empty={empty}  markdown-fence={fence}  <5chars={bad}")
PY
    else
        echo "  [sanity] $name: NO samples/results file found" | tee -a "$LOG"
    fi

    # Teardown server
    kill $SPID 2>/dev/null || true
    wait $SPID 2>/dev/null || true
    sleep 2

    # Delete Q6_K to free disk (keep HF for merge)
    rm -f "$q6k" && echo "  removed Q6_K" | tee -a "$LOG"

    df -h / | tail -1 | tee -a "$LOG"
done

echo | tee -a "$LOG"
echo "===== $(date) ALL SOURCE EVALS DONE =====" | tee -a "$LOG"
echo "Results:" | tee -a "$LOG"
for m in "${MODELS[@]}"; do
    name=$(basename "$m")
    result_file=$(ls $WORKDIR/humaneval_${name}/**/results_*.json 2>/dev/null | head -1 || true)
    if [[ -n "$result_file" ]]; then
        score=$(python3 -c "import json; d = json.load(open('$result_file')); r = list(d['results'].values())[0]; print(next((f'{v:.4f}' for k,v in r.items() if 'pass' in k.lower() and 'stderr' not in k), 'N/A'))")
        echo "  $name: pass@1 = $score" | tee -a "$LOG"
    else
        echo "  $name: NO RESULT" | tee -a "$LOG"
    fi
done
