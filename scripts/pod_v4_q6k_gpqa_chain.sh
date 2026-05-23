#!/bin/bash
# pod_v4_q6k_gpqa_chain.sh — GPQA Diamond 198q comparison: standard v4 Q6_K vs MTP v4 Q6_K.
#
# Pre-conditions:
#   - MTP rebuild via quantize_gguf.py has completed; MTP Q6_K is at
#     /workspace/out/Qwen3.6-27B-Omnimerge-v4-MTP-GGUF/Qwen3.6-27B-Omnimerge-v4-Q6_K.gguf
#     (or wherever quantize_gguf.py landed it).
#   - llama.cpp built at /opt/llama.cpp/build/bin/ (pod-installed by quantize_gguf --pod).
#   - HF_TOKEN exported.
#   - Python3 + lm_eval available (pod_setup installs huggingface_hub + transformers;
#     lm_eval[api] gets installed here lazily if missing).
#
# Phase A: standard v4 Q6_K (download from ManniX-ITA/Qwen3.6-27B-Omnimerge-v4-GGUF).
#   This fills the partial 177q gap on the published v4-GGUF card with a clean 198q number.
# Phase B: MTP v4 Q6_K (just built locally) with --spec-type draft-mtp.
#   This validates quality parity (same body weights) + measures speculative-decoding speedup.
#
# Quality is the headline. Speedup is measured by wall-time delta between the two runs at
# identical --parallel/concurrency settings. Both runs use --use_cache + --log_samples per
# project rule; sqlite cache lives next to the result dir so any retry resumes cleanly.

set -uo pipefail

WORK=/workspace
EVAL_ROOT=$WORK/eval_results
SCRIPT_DIR=$WORK/scripts
LOG_DIR=$WORK/logs

STD_GGUF_URL="https://huggingface.co/ManniX-ITA/Qwen3.6-27B-Omnimerge-v4-GGUF/resolve/main/Qwen3.6-27B-Omnimerge-v4-Q6_K.gguf"
STD_GGUF=$WORK/out_standard/Qwen3.6-27B-Omnimerge-v4-Q6_K.gguf
MTP_GGUF=$WORK/out/Qwen3.6-27B-Omnimerge-v4-Q6_K.gguf
TOKENIZER="ManniX-ITA/Qwen3.6-27B-Omnimerge-v4"

PORT=8099
SERVED_NAME_STD="qwen36-v4-q6k-std"
SERVED_NAME_MTP="qwen36-v4-q6k-mtp"

mkdir -p $EVAL_ROOT $WORK/out_standard $LOG_DIR
TS=$(date +%Y%m%d_%H%M%S)

# ── 0. Pre-flight: confirm MTP rebuild is done + MTP Q6_K exists ─────────
echo "[$(date +%H:%M:%S)] ===== Phase 0: pre-flight ====="
if [ ! -f "$MTP_GGUF" ]; then
    echo "ERROR: $MTP_GGUF missing — MTP rebuild may still be running or failed."
    echo "Check /workspace/logs/mtp_rebuild_*.log and /workspace/out/ contents."
    ls -la $WORK/out/ 2>/dev/null || echo "  (out dir empty)"
    exit 1
fi
echo "  MTP Q6_K present: $(du -h $MTP_GGUF | cut -f1)"

# Sanity: confirm MTP head landed at blk.{N}.* via llama-gguf
LLAMA_GGUF=/opt/llama.cpp/build/bin/llama-gguf
if [ -x "$LLAMA_GGUF" ]; then
    echo "  inspecting MTP Q6_K for blk.N.* (MTP head)..."
    $LLAMA_GGUF r "$MTP_GGUF" 2>&1 | grep -E "blk\.6[0-9]\." | head -5 || echo "    (no MTP-block prefix found — speculative decoding will fail)"
fi

# Ensure lm_eval available
if ! command -v lm_eval >/dev/null 2>&1; then
    echo "  installing lm-eval[api]==0.4.11 ..."
    pip install --quiet 'lm-eval[api]==0.4.11' 2>&1 | tail -3
fi

# ── 1. Download standard v4 Q6_K from existing -GGUF repo ────────────────
echo
echo "[$(date +%H:%M:%S)] ===== Phase 1: download standard v4 Q6_K ====="
if [ ! -f "$STD_GGUF" ]; then
    echo "  downloading $(basename $STD_GGUF) (22.1 GB)..."
    curl -L --fail --retry 5 --retry-delay 10 -H "Authorization: Bearer ${HF_TOKEN}" \
        -o "$STD_GGUF" "$STD_GGUF_URL" 2>&1 | tail -3
    if [ ! -s "$STD_GGUF" ]; then
        echo "ERROR: download failed."
        exit 1
    fi
fi
echo "  standard Q6_K: $(du -h $STD_GGUF | cut -f1)"

# ── 2. Common helper: run llama-server + lm_eval + collect ───────────────
run_gpqa() {
    local label="$1"
    local gguf="$2"
    local extra_args="$3"      # e.g. "--spec-type draft-mtp --spec-draft-n-max 3"
    local served_name="$4"

    local result_dir=$EVAL_ROOT/$label
    local cache_dir=$result_dir/sqlite_cache
    local server_log=$LOG_DIR/${label}_llama_server_${TS}.log
    local eval_log=$LOG_DIR/${label}_lm_eval_${TS}.log
    mkdir -p "$result_dir" "$cache_dir"

    echo
    echo "[$(date +%H:%M:%S)] ===== $label : launching llama-server ====="
    echo "  GGUF: $gguf"
    echo "  extra args: $extra_args"
    echo "  result_dir: $result_dir"
    echo "  cache_dir:  $cache_dir"
    echo "  server log: $server_log"

    # llama-server with the canonical 27B GPQA recipe: --parallel 2,
    # q8 K/V cache, deepseek reasoning, 8k reasoning budget.
    /opt/llama.cpp/build/bin/llama-server \
        -m "$gguf" \
        --port $PORT \
        -c 65536 -ngl 99 \
        --parallel 2 \
        --cache-type-k q8_0 --cache-type-v q8_0 \
        --reasoning-format deepseek --reasoning-budget 8192 \
        --no-warmup \
        --alias "$served_name" \
        $extra_args \
        > "$server_log" 2>&1 &
    local SERVER_PID=$!
    disown $SERVER_PID
    echo "  server PID=$SERVER_PID"

    # Wait until /v1/models responds
    echo -n "  waiting for server "
    for i in $(seq 1 120); do
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/v1/models 2>/dev/null | grep -q "200"; then
            echo " ready (after ${i}s)"
            break
        fi
        sleep 2
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo
            echo "  ERROR: llama-server died during boot. Last 20 lines of $server_log:"
            tail -20 "$server_log"
            return 1
        fi
        printf "."
    done

    # Tight probe — one curl chat completion
    PROBE=$(curl -s -X POST http://localhost:$PORT/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$served_name\",\"prompt\":\"The capital of France is\",\"max_tokens\":10,\"temperature\":1.0,\"top_p\":0.95}" 2>&1)
    echo "  probe response (first 200 chars): $(echo $PROBE | head -c 200)"

    # ── 3. lm_eval: GPQA Diamond 198q, raw completions, --use_cache, --log_samples
    echo
    echo "[$(date +%H:%M:%S)] ===== $label : lm_eval GPQA Diamond 198q ====="
    local t0=$(date +%s)
    lm_eval \
        --model local-completions \
        --model_args "model=$served_name,base_url=http://localhost:$PORT/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=$TOKENIZER,max_gen_toks=8192,max_retries=8,timeout=1800" \
        --tasks gpqa_diamond_cot_zeroshot \
        --batch_size 1 \
        --use_cache "$cache_dir/${label}" \
        --log_samples \
        --output_path "$result_dir" \
        2>&1 | tee "$eval_log"
    local rc=${PIPESTATUS[0]}
    local elapsed=$(($(date +%s) - t0))
    echo
    echo "[$(date +%H:%M:%S)] $label : lm_eval exit=$rc elapsed=${elapsed}s"

    # Kill the server cleanly
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null

    # Save the elapsed time for the speedup comparison
    echo "$elapsed" > "$result_dir/wall_time_seconds.txt"

    # Sanity: count samples
    local samples=$(find "$result_dir" -name "samples_gpqa*.jsonl" | head -1)
    if [ -n "$samples" ]; then
        local n=$(wc -l < "$samples")
        echo "  samples written: $n (expected 198)"
    fi

    return $rc
}

# ── Phase A: standard v4 Q6_K ────────────────────────────────────────────
run_gpqa "v4_q6k_standard" "$STD_GGUF" "" "$SERVED_NAME_STD"
rc_a=$?

# Free disk: keep MTP Q6_K, ditch standard so MTP run has headroom (if needed)
# Standard is 22 GB; we don't need it for the MTP phase. Don't delete yet — user
# may want to re-run from sqlite cache if anything looks off.

# ── Phase B: MTP v4 Q6_K with speculative decoding ───────────────────────
run_gpqa "v4_q6k_mtp" "$MTP_GGUF" "--spec-type draft-mtp --spec-draft-n-max 3" "$SERVED_NAME_MTP"
rc_b=$?

# ── Summary ──────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo "  GPQA Diamond 198q comparison — summary"
echo "============================================================"
for label in v4_q6k_standard v4_q6k_mtp; do
    res=$(find "$EVAL_ROOT/$label" -name "results_*.json" | head -1)
    elapsed=$(cat "$EVAL_ROOT/$label/wall_time_seconds.txt" 2>/dev/null || echo "?")
    if [ -n "$res" ]; then
        flex=$(python3 -c "
import json
d = json.load(open('$res'))
r = d.get('results', {}).get('gpqa_diamond_cot_zeroshot', {})
print(f\"flex={r.get('exact_match,flexible-extract', '?'):.4f}  strict={r.get('exact_match,strict-match', '?'):.4f}\" if r else '(no results yet)')
" 2>/dev/null || echo "(could not parse)")
        echo "  $label : $flex  wall=${elapsed}s"
    else
        echo "  $label : (no results JSON yet)"
    fi
done
echo "============================================================"
echo "  rc_standard=$rc_a  rc_mtp=$rc_b"
echo "  results dirs: $EVAL_ROOT/v4_q6k_{standard,mtp}/"
echo "  sqlite caches: $EVAL_ROOT/v4_q6k_{standard,mtp}/sqlite_cache/"
echo "============================================================"
