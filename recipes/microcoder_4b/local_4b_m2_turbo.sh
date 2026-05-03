#!/bin/bash
# M2-turbo: identical to M2 OMv2 (no importance signal) but with the new
# --pr682-turbo flag. Single-variable A/B vs M2.
#
# Reference M2 (published): HE 52.44 / MBPP 49.40 (Q6_K, parallel-2, raw /v1/completions)
set -uo pipefail

WORKSPACE=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
HF_DIR="$WORKSPACE/hf_models_4b"
OUT="$WORKSPACE/4b_phase1"
LOGS="$WORKSPACE/logs"
LLAMA=/opt/llama.cpp/build/bin
PYBIN=/shared/dev/lightseek/.venv/bin/python  # has torch + safetensors

BASE="$HF_DIR/Qwen3.5-4B"
SRC1="$HF_DIR/jackrong-v2"
SRC2="$HF_DIR/crow-4b"

NAME=m2_turbo
M_DIR="$OUT/merged/${NAME}"
M_F16="$OUT/gguf/${NAME}-F16.gguf"
M_Q6K="$OUT/gguf/${NAME}-Q6_K.gguf"
LOG_MASTER="$LOGS/4b_${NAME}.log"

mkdir -p "$M_DIR" "$OUT/gguf" "$LOGS"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1

echo "=== ${NAME} starting $(date -Iseconds) ===" | tee -a "$LOG_MASTER"

# --- Step 1: merge with M2 settings + --pr682-turbo --------------
if [ ! -f "$M_DIR/.merge_done" ] || ! find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
    echo "[*] running dare_ties_merge.py omnimerge_v2 + protect-critical-layers" | tee -a "$LOG_MASTER"
    rm -rf "$M_DIR"
    mkdir -p "$M_DIR"
    "$PYBIN" -u "$WORKSPACE/scripts/dare_ties_merge.py" \
        --base "$BASE" \
        --source "$SRC1" --source "$SRC2" \
        --output "$M_DIR" \
        --method omnimerge_v2 \
        --v2-features obim,darex,emr \
        --weights 0.55,0.45 \
        --density 0.53 \
        --darex-q 0.75 \
        --pr682-turbo \
        --seed 42 \
        --device cuda \
        2>&1 | tee -a "$LOGS/4b_merge_${NAME}.log"
    if find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
        touch "$M_DIR/.merge_done"
    else
        echo "[!] merge produced no safetensors — aborting" | tee -a "$LOG_MASTER"
        exit 1
    fi
else
    echo "[*] merge: SKIP (already done)" | tee -a "$LOG_MASTER"
fi

# --- Step 2: convert + quantize ----------------------------------------------
if [ ! -f "$M_Q6K" ]; then
    echo "[*] convert HF → F16 GGUF" | tee -a "$LOG_MASTER"
    python3 /opt/llama.cpp/convert_hf_to_gguf.py "$M_DIR" --outtype f16 --outfile "$M_F16" \
        2>&1 | tee -a "$LOGS/4b_convert_${NAME}.log" | tail -10
    echo "[*] quantize Q6_K" | tee -a "$LOG_MASTER"
    "$LLAMA/llama-quantize" "$M_F16" "$M_Q6K" Q6_K 2>&1 | tee -a "$LOGS/4b_quantize_${NAME}.log" | tail -5
    rm -f "$M_F16"
else
    echo "[*] quantize: SKIP" | tee -a "$LOG_MASTER"
fi

# --- Step 3: launch llama-server ---------------------------------------------
pkill -f 'llama-server.*--port 8099' 2>/dev/null
sleep 2
nohup "$LLAMA/llama-server" \
    -m "$M_Q6K" --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
    --reasoning-format deepseek --reasoning-budget 8192 \
    --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 \
    > "$LOGS/4b_server_${NAME}.log" 2>&1 &
SERVER_PID=$!
disown
for i in $(seq 1 60); do
    curl -sf http://localhost:8099/health > /dev/null 2>&1 && break
    sleep 2
done
curl -sf http://localhost:8099/health > /dev/null 2>&1 || { echo "[!] server failed"; tail -30 "$LOGS/4b_server_${NAME}.log"; exit 1; }
echo "    server ready (PID $SERVER_PID)" | tee -a "$LOG_MASTER"

# --- Step 4: HE + MBPP eval --------------------------------------------------
EVAL_DIR="$OUT/eval_results"
for task in mbpp humaneval; do
    echo "[*] $task eval $(date -Iseconds)" | tee -a "$LOG_MASTER"
    OUTPATH="$EVAL_DIR/${task}_${NAME}"
    mkdir -p "$OUTPATH"
    /shared/dev/lightseek/.venv/bin/python -u -m lm_eval \
        --model local-completions \
        --model_args "model=${NAME},base_url=http://localhost:8099/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=${BASE},max_gen_toks=2048" \
        --tasks "$task" \
        --gen_kwargs "temperature=0.0,top_p=1.0" \
        --batch_size 1 --use_cache "$OUTPATH/cache" \
        --log_samples --confirm_run_unsafe_code \
        --output_path "$OUTPATH" \
        2>&1 | tee -a "$LOGS/4b_${task}_${NAME}.log" | tail -15
done

kill "$SERVER_PID" 2>/dev/null
sleep 2
echo "=== ${NAME} done $(date -Iseconds) ===" | tee -a "$LOG_MASTER"
find "$EVAL_DIR" -path "*${NAME}*" -name "results_*.json" 2>/dev/null
