#!/bin/bash
# Eval 3 community Qwen3.5-4B coder fine-tunes + base on HE/MBPP/LCB-Medium-30.
# Same methodology as M1..M8 study: parallel-2 llama-server, raw /v1/completions
# for HE/MBPP, /v1/chat/completions for LCB. Q6_K quants.
set -uo pipefail

WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
HF=$WS/hf_models_4b
COD=$HF/coder_eval
OUT=$WS/4b_phase1
GGUF_DIR=$OUT/gguf_coder
EVAL_DIR=$OUT/eval_results
LOGS=$WS/logs
LLAMA=/opt/llama.cpp/build/bin
PYBIN=/shared/dev/lightseek/.venv/bin/python
TOK_BASE=$HF/Qwen3.5-4B  # tokenizer for HE/MBPP raw completions

mkdir -p "$GGUF_DIR" "$EVAL_DIR" "$LOGS"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1

# --- normalize jackrong shard filenames: hf gives `model.safetensors-NNNNN-of-NNNNN.safetensors`
# convert_hf_to_gguf expects `model-NNNNN-of-NNNNN.safetensors` — rename + rewrite index.
fix_jackrong() {
    local D=$COD/jackrong-python
    if [ -f "$D/model.safetensors-00001-of-00002.safetensors" ]; then
        echo "[fix] renaming jackrong shards"
        mv "$D/model.safetensors-00001-of-00002.safetensors" "$D/model-00001-of-00002.safetensors"
        mv "$D/model.safetensors-00002-of-00002.safetensors" "$D/model-00002-of-00002.safetensors"
        sed -i 's/"model\.safetensors-/"model-/g' "$D/model.safetensors.index.json"
    fi
}

convert_to_q6k() {
    local SRC=$1 NAME=$2
    local F16=$GGUF_DIR/${NAME}.F16.gguf
    local Q6K=$GGUF_DIR/${NAME}.Q6_K.gguf
    if [ -f "$Q6K" ]; then echo "[$NAME] Q6_K already exists"; return 0; fi
    echo "[$NAME] convert hf -> F16"
    python3 /opt/llama.cpp/convert_hf_to_gguf.py "$SRC" --outtype f16 --outfile "$F16" 2>&1 \
        | tee "$LOGS/conv_${NAME}.log" | tail -10
    [ -f "$F16" ] || { echo "[$NAME] F16 failed"; return 1; }
    echo "[$NAME] quantize -> Q6_K"
    "$LLAMA/llama-quantize" "$F16" "$Q6K" Q6_K 2>&1 \
        | tee "$LOGS/quant_${NAME}.log" | tail -3
    [ -f "$Q6K" ] || { echo "[$NAME] Q6_K failed"; return 1; }
    rm -f "$F16"
    echo "[$NAME] Q6_K done: $(ls -lh $Q6K | awk '{print $5}')"
}

run_eval_block() {
    local NAME=$1 GGUF=$2 TOK=$3
    echo "=== $NAME starting $(date -Iseconds) ==="
    pkill -f 'llama-server.*--port 8099' 2>/dev/null
    sleep 2
    # NB: NO --reasoning-format here. Base Qwen3.5-4B is non-reasoning and
    # the deepseek format made llama-server return empty content (everything
    # eaten as reasoning until a </think> that never came). For LCB chat
    # completions we want the FULL text; clean_lcb_completion regex extracts
    # the fenced code block regardless of any <think> prefix.
    nohup "$LLAMA/llama-server" \
        -m "$GGUF" --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
        --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 \
        > "$LOGS/4b_server_${NAME}.log" 2>&1 &
    SERVER_PID=$!
    disown
    for i in $(seq 1 60); do
        curl -sf http://localhost:8099/health >/dev/null 2>&1 && break
        sleep 2
    done
    curl -sf http://localhost:8099/health >/dev/null 2>&1 || {
        echo "[!] $NAME server failed"; tail -30 "$LOGS/4b_server_${NAME}.log"
        kill $SERVER_PID 2>/dev/null; return 1
    }
    echo "  server ready"

    # HE + MBPP via /v1/completions
    for task in mbpp humaneval; do
        echo "  $task $(date -Iseconds)"
        local OP=$EVAL_DIR/${task}_${NAME}
        mkdir -p "$OP"
        "$PYBIN" -u -m lm_eval \
            --model local-completions \
            --model_args "model=${NAME},base_url=http://localhost:8099/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=${TOK},max_gen_toks=2048" \
            --tasks "$task" \
            --gen_kwargs "temperature=0.0,top_p=1.0" \
            --batch_size 1 --use_cache "$OP/cache" \
            --log_samples --confirm_run_unsafe_code \
            --output_path "$OP" \
            2>&1 | tee "$LOGS/4b_${task}_${NAME}.log" | tail -8
    done

    # LCB-Medium-30 via /v1/chat/completions
    echo "  lcb $(date -Iseconds)"
    local LP=$EVAL_DIR/lcb_${NAME}
    if [ -f "$LP/lcb_results.json" ]; then
        local PA1=$(python3 -c "import json; print(json.load(open('$LP/lcb_results.json'))['pass_at_1'])" 2>/dev/null)
        if [ -n "$PA1" ] && [ "$PA1" != "0.0" ]; then
            echo "  lcb already valid: pass@1=$PA1 (skipping)"
            kill $SERVER_PID 2>/dev/null
            sleep 2
            echo "=== $NAME done $(date -Iseconds) ==="
            return 0
        fi
        echo "  lcb prior result is 0.0 — redoing"
        rm -f "$LP/lcb_results.json" "$LP/lcb_samples.jsonl"
    fi
    mkdir -p "$LP"
    "$PYBIN" -u "$WS/scripts/lcb_llama_server.py" \
        --name "$NAME" --base-url http://localhost:8099 \
        --limit 30 --difficulty medium --min-date 2024-10-01 \
        --max-tokens 2048 \
        --output "$LP/lcb_results.json" \
        --samples "$LP/lcb_samples.jsonl" \
        2>&1 | tee "$LOGS/4b_lcb_${NAME}.log" | tail -40

    kill $SERVER_PID 2>/dev/null
    sleep 2
    echo "=== $NAME done $(date -Iseconds) ==="
}

# --- prep ---
fix_jackrong
convert_to_q6k "$COD/continuum-128k-forged" "continuum-128k-forged" || exit 1
convert_to_q6k "$COD/jackrong-python"        "jackrong-python"        || exit 1
[ -f "$COD/massivdash-ts/Qwen3.5-4B.Q6_K.gguf" ] && \
    cp -n "$COD/massivdash-ts/Qwen3.5-4B.Q6_K.gguf" "$GGUF_DIR/massivdash-ts.Q6_K.gguf"

# --- evals ---
# base LCB only (HE+MBPP already evaluated previously, but redo for consistency)
run_eval_block "qwen3.5-4b-base"        "$OUT/gguf_base/Qwen_Qwen3.5-4B-Q6_K.gguf"   "$TOK_BASE"
run_eval_block "continuum-128k-forged"  "$GGUF_DIR/continuum-128k-forged.Q6_K.gguf"  "$TOK_BASE"
run_eval_block "jackrong-python"        "$GGUF_DIR/jackrong-python.Q6_K.gguf"        "$TOK_BASE"
run_eval_block "massivdash-ts"          "$GGUF_DIR/massivdash-ts.Q6_K.gguf"          "$TOK_BASE"

echo "=== all coder evals done $(date -Iseconds) ==="
