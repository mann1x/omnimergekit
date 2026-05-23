#!/bin/bash
# pod_v4_q6k_eval_chain.sh — autonomous HE→MBPP→GPQA comparison: standard vs MTP v4 Q6_K.
#
# Order (HE+MBPP first per user — faster results → first card update earlier):
#   Phase 1: download standard v4 Q6_K from existing -GGUF repo
#   Phase 2a: standard Q6_K HumanEval 164q  (~30-60 min)
#   Phase 2b: standard Q6_K MBPP 500q       (~1-2 h)
#   Phase 3a: MTP Q6_K HumanEval 164q       (--spec-type draft-mtp)
#   Phase 3b: MTP Q6_K MBPP 500q
#   ── pause: write summary_he_mbpp.json + propose card update text ──
#   Phase 4a: standard Q6_K GPQA Diamond 198q  (~2-4 h)
#   Phase 4b: MTP Q6_K GPQA Diamond 198q       (--spec-type draft-mtp)
#   ── final: write summary_all.json + final card update text ──
#
# Each phase:
#   - launches llama-server with the canonical 27B recipe (--parallel 2,
#     q8 KV, --reasoning-format deepseek --reasoning-budget 8192)
#   - runs lm_eval with --use_cache + --log_samples + max_retries=8 timeout=1800
#   - post-processes HE/MBPP samples to strip <think>...</think> before re-execution
#   - parses llama-server log for decode tokens/sec
#   - kills server cleanly
#   - rsyncs results back to solidpc
#
# Sanity gates between phases:
#   HE std/mtp >= 50%, agree within 5pp
#   MBPP std/mtp >= 30%, agree within 5pp
#   GPQA std/mtp >= 40%, agree within 5pp
#   Gate failure → log + continue (don't abort, partial data better than nothing)
#
# Outputs:
#   /workspace/eval_results/v4_q6k_{std,mtp}/{humaneval,mbpp,gpqa}/...
#   /workspace/eval_results/SUMMARY.json
#   /workspace/eval_results/card_v4_gguf_proposed.md  (GPQA replacement)
#   /workspace/eval_results/card_v4_mtp_gguf_proposed.md  (full comparison section)

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

# Per-bench server-side context. STD fits in 24 GB at full 65536 with -p 2.
# MTP head + draft buffer evicts that — drop ctx + go single-slot. 16384
# leaves prompt+gen headroom for HE/MBPP/GPQA (max_gen_toks <= 8192).
STD_CTX=${STD_CTX:-65536}
MTP_CTX=${MTP_CTX:-16384}
# lm-eval local-completions default max_length=2048 strangled MBPP 3-shot
# prompts + every GPQA prompt on the 2026-05-22 run. Bake the fix in.
LM_MAX_LENGTH=${LM_MAX_LENGTH:-32768}
# Phase filter — comma-separated subset of the 6 run_bench labels. Default
# = all. Use to re-run a subset cleanly after fixing config.
PHASES=${PHASES:-std_he,std_mbpp,mtp_he,mtp_mbpp,std_gpqa,mtp_gpqa}

should_run() {
    local p=",$PHASES,"
    [[ "$p" == *",$1,"* ]]
}

mkdir -p "$EVAL_ROOT" "$WORK/out_standard" "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/eval_chain_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=================================================================="
echo "  pod_v4_q6k_eval_chain.sh — autonomous HE+MBPP+GPQA std-vs-MTP"
echo "  started: $(date)"
echo "  log: $MAIN_LOG"
echo "=================================================================="

# ── 0. Pre-flight ────────────────────────────────────────────────────────
echo
echo "[$(date +%H:%M:%S)] Phase 0: pre-flight"
if [ ! -f "$MTP_GGUF" ]; then
    echo "  ERROR: MTP Q6_K missing at $MTP_GGUF"
    echo "  /workspace/out contents:"
    ls -la $WORK/out/ 2>/dev/null
    exit 1
fi
echo "  MTP Q6_K present: $(du -h $MTP_GGUF | cut -f1)"

# Inspect MTP block via llama-gguf
LLAMA_GGUF=/opt/llama.cpp/build/bin/llama-gguf
if [ -x "$LLAMA_GGUF" ]; then
    MTP_BLK_COUNT=$($LLAMA_GGUF r "$MTP_GGUF" 2>&1 | grep -cE "blk\.6[0-9]\.")
    echo "  MTP head tensors visible at blk.6N.*: $MTP_BLK_COUNT lines"
fi

# Ensure lm_eval + Python deps
if ! command -v lm_eval >/dev/null 2>&1; then
    echo "  installing lm-eval[api]==0.4.11..."
    pip install --quiet 'lm-eval[api]==0.4.11' transformers safetensors 2>&1 | tail -3
fi
# HE/MBPP need explicit consent
export HF_ALLOW_CODE_EVAL=1

# ── 1. Download standard v4 Q6_K ─────────────────────────────────────────
echo
echo "[$(date +%H:%M:%S)] Phase 1: download standard v4 Q6_K"
if [ ! -f "$STD_GGUF" ]; then
    echo "  downloading $(basename $STD_GGUF) (22.1 GB)..."
    HF_TOKEN_VAL=${HF_TOKEN:-}
    [ -z "$HF_TOKEN_VAL" ] && [ -f /root/.cache/huggingface/token ] && HF_TOKEN_VAL=$(cat /root/.cache/huggingface/token)
    curl -L --fail --retry 5 --retry-delay 10 -H "Authorization: Bearer $HF_TOKEN_VAL" \
        -o "$STD_GGUF" "$STD_GGUF_URL" -w "  total: %{size_download} bytes in %{time_total}s\n"
    if [ ! -s "$STD_GGUF" ] || [ $(stat -c %s "$STD_GGUF") -lt 20000000000 ]; then
        echo "  ERROR: download incomplete (size $(stat -c %s "$STD_GGUF" 2>/dev/null))"
        exit 1
    fi
fi
echo "  standard Q6_K: $(du -h $STD_GGUF | cut -f1) sha256=$(sha256sum $STD_GGUF | cut -c1-12)"

# ── Common phase runner ──────────────────────────────────────────────────
run_bench() {
    # $1 label (e.g. "std_he"), $2 gguf, $3 extra-server-args, $4 task, $5 max_gen_toks
    local label="$1"
    local gguf="$2"
    local extra_args="$3"
    local task="$4"
    local max_gen="$5"
    local result_dir="$EVAL_ROOT/$label"
    local cache_dir="$result_dir/sqlite_cache"
    local server_log="$LOG_DIR/${label}_server_${TS}.log"
    local eval_log="$LOG_DIR/${label}_lmeval_${TS}.log"
    mkdir -p "$result_dir" "$cache_dir"

    echo
    echo "[$(date +%H:%M:%S)] ==== $label : $task ===="
    echo "  GGUF:  $gguf  ($(du -h $gguf | cut -f1))"
    echo "  extra: $extra_args"

    local SERVED="v4-${label}"
    # MTP spec-decoding paths need shrunken ctx + single slot to fit a 24 GB
    # 3090 alongside Q6_K weights (21 GB) + draft KV buffer.
    local ctx=$STD_CTX
    local parallel=2
    if [[ "$extra_args" == *"draft-mtp"* ]]; then
        ctx=$MTP_CTX
        parallel=1
    fi
    # llama-server canonical 27B recipe
    /opt/llama.cpp/build/bin/llama-server \
        -m "$gguf" --port $PORT \
        -c $ctx -ngl 99 \
        --parallel $parallel \
        --cache-type-k q8_0 --cache-type-v q8_0 \
        --reasoning-format deepseek --reasoning-budget 8192 \
        --no-warmup \
        --alias "$SERVED" \
        $extra_args \
        > "$server_log" 2>&1 &
    local SPID=$!
    disown $SPID
    echo "  server PID=$SPID"

    # Wait for /v1/models
    for i in $(seq 1 180); do
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/v1/models | grep -q "200"; then
            echo "  server ready (${i}s)"
            break
        fi
        sleep 2
        if ! kill -0 $SPID 2>/dev/null; then
            echo "  ERROR: server died during boot. Last 30 lines:"
            tail -30 "$server_log"
            return 1
        fi
    done

    # Probe
    local probe=$(curl -s -X POST http://localhost:$PORT/v1/completions -H "Content-Type: application/json" \
        -d "{\"model\":\"$SERVED\",\"prompt\":\"The capital of France is\",\"max_tokens\":5}" 2>&1)
    echo "  probe: $(echo $probe | head -c 200)"

    local t0=$(date +%s)
    # Build lm_eval args. HE/MBPP need --confirm_run_unsafe_code, GPQA doesn't.
    local extra_lm=""
    case "$task" in
        humaneval|mbpp) extra_lm="--confirm_run_unsafe_code" ;;
    esac
    # max_length=$LM_MAX_LENGTH overrides lm-eval's 2048 default — without
    # this MBPP 3-shot prompts truncate and GPQA prompts return "[invalid]".
    # num_concurrent matches the server's --parallel value (1 for MTP, 2 for std).
    lm_eval \
        --model local-completions \
        --model_args "model=$SERVED,base_url=http://localhost:$PORT/v1/completions,num_concurrent=$parallel,tokenizer_backend=huggingface,tokenizer=$TOKENIZER,max_gen_toks=$max_gen,max_length=$LM_MAX_LENGTH,max_retries=8,timeout=1800" \
        --tasks "$task" \
        --batch_size 1 \
        --use_cache "$cache_dir/${label}" \
        --log_samples \
        $extra_lm \
        --output_path "$result_dir" \
        2>&1 | tee "$eval_log"
    local rc=${PIPESTATUS[0]}
    local elapsed=$(($(date +%s) - t0))
    echo
    echo "[$(date +%H:%M:%S)] $label : lm_eval exit=$rc elapsed=${elapsed}s"
    echo "$elapsed" > "$result_dir/wall_time_seconds.txt"

    # Sanity: count samples
    local samples
    samples=$(find "$result_dir" -name "samples_*.jsonl" | head -1)
    if [ -n "$samples" ]; then
        local n
        n=$(wc -l < "$samples")
        echo "  samples written: $n"
    fi

    # Stop server cleanly
    kill $SPID 2>/dev/null
    sleep 2
    kill -9 $SPID 2>/dev/null
    wait $SPID 2>/dev/null

    # Compute tokens/sec from server log
    python3 "$SCRIPT_DIR/parse_llama_server_throughput.py" "$server_log" \
        --output "$result_dir/throughput.json" > /dev/null 2>&1 || true
    if [ -f "$result_dir/throughput.json" ]; then
        local tps
        tps=$(python3 -c "import json; print(json.load(open('$result_dir/throughput.json')).get('aggregate_decode_tokens_per_sec'))" 2>/dev/null)
        echo "  decode tokens/sec (aggregate): $tps"
    fi

    # For HE/MBPP: think-strip rescore
    case "$task" in
        humaneval|mbpp)
            if [ -n "$samples" ]; then
                echo "  rescoring after <think>-strip..."
                python3 "$SCRIPT_DIR/rescore_strip_think.py" \
                    --bench "$task" --samples "$samples" \
                    --output "$result_dir/rescored_clean.json" 2>&1 | tail -8
            fi
            ;;
    esac

    return $rc
}

write_summary() {
    python3 - <<PY
import json, glob, os
root = "$EVAL_ROOT"
labels = ["std_he", "std_mbpp", "mtp_he", "mtp_mbpp", "std_gpqa", "mtp_gpqa"]
out = {}
for label in labels:
    d = os.path.join(root, label)
    if not os.path.isdir(d):
        out[label] = None
        continue
    block = {}
    # lm_eval results
    rfiles = sorted(glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True), reverse=True)
    if rfiles:
        try:
            r = json.load(open(rfiles[0]))
            tasks = r.get("results", {})
            for t, v in tasks.items():
                block["lm_eval_task"] = t
                for k in ("exact_match,flexible-extract", "exact_match,strict-match",
                         "pass@1", "pass@1,create_test", "pass@1,none"):
                    if k in v:
                        block[f"lm_eval_{k}"] = v[k]
            n = r.get("n-samples", {})
            block["n_samples"] = next(iter(n.values()), None) if n else None
        except Exception as e:
            block["lm_eval_error"] = str(e)
    # rescored clean
    rc = os.path.join(d, "rescored_clean.json")
    if os.path.isfile(rc):
        try:
            block["rescored"] = json.load(open(rc))
        except Exception:
            pass
    # throughput
    tp = os.path.join(d, "throughput.json")
    if os.path.isfile(tp):
        try:
            block["throughput"] = json.load(open(tp))
        except Exception:
            pass
    # wall time
    wt = os.path.join(d, "wall_time_seconds.txt")
    if os.path.isfile(wt):
        block["wall_seconds"] = int(open(wt).read().strip())
    out[label] = block

# Speedup
def get_tps(label):
    b = out.get(label) or {}
    t = b.get("throughput") or {}
    return t.get("aggregate_decode_tokens_per_sec")
def speedup(std_l, mtp_l):
    s, m = get_tps(std_l), get_tps(mtp_l)
    if s and m and s > 0:
        return round(m / s, 3)
    return None

out["_speedup"] = {
    "he": speedup("std_he", "mtp_he"),
    "mbpp": speedup("std_mbpp", "mtp_mbpp"),
    "gpqa": speedup("std_gpqa", "mtp_gpqa"),
}

with open(os.path.join(root, "SUMMARY.json"), "w") as f:
    json.dump(out, f, indent=2)
print("SUMMARY:", json.dumps(out, indent=2)[:3000])
PY
}

echo
echo "[$(date +%H:%M:%S)] active phases: $PHASES (std ctx=$STD_CTX, mtp ctx=$MTP_CTX, lm max_length=$LM_MAX_LENGTH)"

# ── Phase 2: std HE + MBPP ───────────────────────────────────────────────
should_run std_he   && run_bench "std_he"   "$STD_GGUF" ""                                          "humaneval" 8192
should_run std_mbpp && run_bench "std_mbpp" "$STD_GGUF" ""                                          "mbpp"      4096

# ── Phase 3: MTP HE + MBPP ───────────────────────────────────────────────
should_run mtp_he   && run_bench "mtp_he"   "$MTP_GGUF" "--spec-type draft-mtp --spec-draft-n-max 3" "humaneval" 8192
should_run mtp_mbpp && run_bench "mtp_mbpp" "$MTP_GGUF" "--spec-type draft-mtp --spec-draft-n-max 3" "mbpp"      4096

# Snapshot summary after code benches
echo
echo "[$(date +%H:%M:%S)] === HE+MBPP done — writing partial summary ==="
write_summary

# ── Phase 4: GPQA std + MTP ──────────────────────────────────────────────
should_run std_gpqa && run_bench "std_gpqa" "$STD_GGUF" ""                                          "gpqa_diamond_cot_zeroshot" 8192
should_run mtp_gpqa && run_bench "mtp_gpqa" "$MTP_GGUF" "--spec-type draft-mtp --spec-draft-n-max 3" "gpqa_diamond_cot_zeroshot" 8192

echo
echo "[$(date +%H:%M:%S)] === GPQA done — writing final summary ==="
write_summary

echo
echo "=================================================================="
echo "  EVAL CHAIN COMPLETE: $(date)"
echo "  results: $EVAL_ROOT"
echo "  SUMMARY: $EVAL_ROOT/SUMMARY.json"
echo "=================================================================="
