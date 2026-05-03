#!/bin/bash
# Omnimerge v2 experiment on pod 34628028
# Run AFTER 98e v3 GGUF + Omnimerge missing quants are done.
# Pipeline: download sources → merge v2 → convert GGUF → quantize Q6_K → eval MBPP → eval HumanEval → eval GPQA
set -euo pipefail

WORKSPACE=/workspace
BASE_MODEL="Qwen/Qwen3.5-27B"
SOURCE_CLAUDE="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"
SOURCE_ESPER="ValiantLabs/Qwen3.5-27B-Esper3.1"
SOURCE_GEMINI="Jackrong/Qwen3.5-27B-Gemini-3.1-Pro-Reasoning-Distill"
OUTPUT_DIR="$WORKSPACE/omnimerge-v2"
GGUF_DIR="$WORKSPACE/gguf_v2"
LOG="$WORKSPACE/omnimerge_v2.log"

# Merge parameters
METHOD="omnimerge_v2"
DENSITY="0.53"
WEIGHTS="0.40,0.35,0.25"
DAREX_Q="0.75"
SEED=42

export HF_ALLOW_CODE_EVAL=1
echo "=== Omnimerge v2 experiment ===" | tee "$LOG"
echo "Started: $(date -Iseconds)" | tee -a "$LOG"

# Step 0: Clean up space if needed
echo "=== Disk check ===" | tee -a "$LOG"
df -h /workspace | tee -a "$LOG"

# Step 1: Download source models (skip if already present)
echo "=== Step 1: Download source models ===" | tee -a "$LOG"
for MODEL in "$BASE_MODEL" "$SOURCE_CLAUDE" "$SOURCE_ESPER" "$SOURCE_GEMINI"; do
    SHORT=$(basename "$MODEL")
    DEST="$WORKSPACE/hf_models/$SHORT"
    if [ -d "$DEST" ] && [ "$(ls "$DEST"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  $SHORT already downloaded, skipping" | tee -a "$LOG"
    else
        echo "  Downloading $MODEL..." | tee -a "$LOG"
        hf download "$MODEL" --local-dir "$DEST" 2>&1 | tail -5 | tee -a "$LOG"
    fi
done

# Step 2: Run merge
echo "=== Step 2: Merge (method=$METHOD, density=$DENSITY, darex_q=$DAREX_Q) ===" | tee -a "$LOG"
python3 /workspace/dare_ties_merge.py \
    --base "$WORKSPACE/hf_models/Qwen3.5-27B" \
    --source "$WORKSPACE/hf_models/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled" \
    --source "$WORKSPACE/hf_models/Qwen3.5-27B-Esper3.1" \
    --source "$WORKSPACE/hf_models/Qwen3.5-27B-Gemini-3.1-Pro-Reasoning-Distill" \
    --output "$OUTPUT_DIR" \
    --method "$METHOD" \
    --density "$DENSITY" \
    --weights "$WEIGHTS" \
    --darex-q "$DAREX_Q" \
    --seed "$SEED" \
    --shard-size 5 \
    2>&1 | tee -a "$LOG"

echo "Merge complete: $(date -Iseconds)" | tee -a "$LOG"

# Step 3: Convert to F16 GGUF
echo "=== Step 3: Convert to F16 GGUF ===" | tee -a "$LOG"
mkdir -p "$GGUF_DIR"
python3 /opt/llama.cpp/convert_hf_to_gguf.py "$OUTPUT_DIR" \
    --outfile "$GGUF_DIR/omnimerge-v2-F16.gguf" \
    --outtype f16 \
    2>&1 | tee -a "$LOG"

# Step 4: Quantize to Q6_K (fast, good quality for eval)
echo "=== Step 4: Quantize Q6_K ===" | tee -a "$LOG"
/opt/llama.cpp/build/bin/llama-quantize \
    "$GGUF_DIR/omnimerge-v2-F16.gguf" \
    "$GGUF_DIR/omnimerge-v2-Q6_K.gguf" \
    Q6_K \
    2>&1 | tee -a "$LOG"

echo "Q6_K ready: $(ls -lh "$GGUF_DIR/omnimerge-v2-Q6_K.gguf")" | tee -a "$LOG"

# Clean F16 to save space
rm -f "$GGUF_DIR/omnimerge-v2-F16.gguf"
echo "F16 deleted" | tee -a "$LOG"

# Step 5: Start llama-server for eval
echo "=== Step 5: Start llama-server ===" | tee -a "$LOG"
pkill -f llama-server 2>/dev/null || true
sleep 2

/opt/llama.cpp/build/bin/llama-server \
    -m "$GGUF_DIR/omnimerge-v2-Q6_K.gguf" \
    --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
    --reasoning-format deepseek --reasoning-budget 8192 \
    > "$WORKSPACE/llama_server_v2.log" 2>&1 &
disown

echo "Waiting for server startup..." | tee -a "$LOG"
for i in $(seq 1 60); do
    if curl -s http://localhost:8099/health | grep -q '"status":"ok"'; then
        echo "  Server ready after ${i}s" | tee -a "$LOG"
        break
    fi
    sleep 2
done

# Step 6: MBPP eval (fastest, ~10 min)
echo "=== Step 6: MBPP eval ===" | tee -a "$LOG"
TOKENIZER="$WORKSPACE/hf_models/Qwen3.5-27B"
CACHE_DIR="$WORKSPACE/eval_cache_v2"
RESULTS_DIR="$WORKSPACE/eval_results_v2"
mkdir -p "$CACHE_DIR" "$RESULTS_DIR"

# Activate lightseek env for lm_eval
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate lightseek 2>/dev/null || true

LM_EVAL=$(which lm_eval 2>/dev/null || echo "/opt/conda/envs/lightseek/bin/lm_eval")

$LM_EVAL \
    --model local-completions \
    --model_args "model=omnimerge-v2,base_url=http://localhost:8099/v1/completions,num_concurrent=1,tokenizer_backend=huggingface,tokenizer=$TOKENIZER,max_length=32768" \
    --tasks mbpp \
    --batch_size 1 \
    --use_cache "$CACHE_DIR/mbpp_omnimerge_v2" \
    --log_samples \
    --output_path "$RESULTS_DIR/mbpp_omnimerge_v2" \
    2>&1 | tee -a "$LOG"

echo "MBPP complete: $(date -Iseconds)" | tee -a "$LOG"
echo "=== MBPP DONE — check results before continuing ===" | tee -a "$LOG"

# Step 7: HumanEval (if MBPP looks good, ~15 min)
echo "=== Step 7: HumanEval eval ===" | tee -a "$LOG"
$LM_EVAL \
    --model local-completions \
    --model_args "model=omnimerge-v2,base_url=http://localhost:8099/v1/completions,num_concurrent=1,tokenizer_backend=huggingface,tokenizer=$TOKENIZER,max_length=32768" \
    --tasks humaneval \
    --batch_size 1 \
    --use_cache "$CACHE_DIR/humaneval_omnimerge_v2" \
    --log_samples \
    --output_path "$RESULTS_DIR/humaneval_omnimerge_v2" \
    2>&1 | tee -a "$LOG"

echo "HumanEval complete: $(date -Iseconds)" | tee -a "$LOG"

# Step 8: GPQA Diamond (long, ~6-10h)
echo "=== Step 8: GPQA Diamond eval ===" | tee -a "$LOG"
# Switch to completions endpoint with reasoning for GPQA (avoids PEG parser crash)
pkill -f llama-server 2>/dev/null || true
sleep 2

/opt/llama.cpp/build/bin/llama-server \
    -m "$GGUF_DIR/omnimerge-v2-Q6_K.gguf" \
    --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
    --reasoning-format deepseek --reasoning-budget 16384 \
    > "$WORKSPACE/llama_server_v2_gpqa.log" 2>&1 &
disown

echo "Waiting for GPQA server startup..." | tee -a "$LOG"
for i in $(seq 1 60); do
    if curl -s http://localhost:8099/health | grep -q '"status":"ok"'; then
        echo "  Server ready after ${i}s" | tee -a "$LOG"
        break
    fi
    sleep 2
done

$LM_EVAL \
    --model local-completions \
    --model_args "model=omnimerge-v2,base_url=http://localhost:8099/v1/completions,num_concurrent=1,tokenizer_backend=huggingface,tokenizer=$TOKENIZER,max_length=32768,max_gen_toks=16384" \
    --tasks gpqa_diamond_cot_zeroshot \
    --apply_chat_template \
    --batch_size 1 \
    --use_cache "$CACHE_DIR/gpqa_omnimerge_v2" \
    --log_samples \
    --output_path "$RESULTS_DIR/gpqa_omnimerge_v2" \
    2>&1 | tee -a "$LOG"

echo "GPQA complete: $(date -Iseconds)" | tee -a "$LOG"
echo "=== ALL EVALS DONE ===" | tee -a "$LOG"
echo "Finished: $(date -Iseconds)" | tee -a "$LOG"

# Summary
echo ""
echo "=== RESULTS SUMMARY ==="
echo "Check: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
