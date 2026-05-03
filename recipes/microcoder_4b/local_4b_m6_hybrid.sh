#!/bin/bash
# M6: hybrid Fisher (attention) + LRP (MLP) routing.
# Same merge config as M4-v2 (w=1/1, d=0.7, density-trimmed via mergekit ex-LRP),
# fed a hybrid importance signal so the layer-type hypothesis can be tested
# without changing any other variable.
set -uo pipefail

WORKSPACE=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
HF_DIR="$WORKSPACE/hf_models_4b"
OUT="$WORKSPACE/4b_phase1"
LOGS="$WORKSPACE/logs"
MERGEKIT_PY=/root/anaconda3/envs/mergekit/bin/python
LLAMA=/opt/llama.cpp/build/bin

BASE="$HF_DIR/Qwen3.5-4B"
SRC1="$HF_DIR/jackrong-v2"
SRC2="$HF_DIR/crow-4b"

LRP1="$HF_DIR/M4-ex-LRP-lrp-scores/lrp/jackrong-v2_lrp_scores.safetensors"
LRP2="$HF_DIR/M4-ex-LRP-lrp-scores/lrp/crow-4b_lrp_scores.safetensors"
F1="$HF_DIR/M3-Fisher-scores/fisher/jackrong-v2_fisher.safetensors"
F2="$HF_DIR/M3-Fisher-scores/fisher/crow-4b_fisher.safetensors"

HYB_DIR="$OUT/hybrid_scores"
HYB1="$HYB_DIR/jackrong-v2_hybrid.safetensors"
HYB2="$HYB_DIR/crow-4b_hybrid.safetensors"

M6_DIR="$OUT/merged/m6_hybrid"
M6_F16="$OUT/gguf/m6_hybrid-F16.gguf"
M6_Q6K="$OUT/gguf/m6_hybrid-Q6_K.gguf"
M6_NAME="m6_hybrid"
LOG_MASTER="$LOGS/4b_m6.log"

mkdir -p "$HYB_DIR" "$OUT/lrp" "$OUT/gguf" "$LOGS"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1

echo "=== M6 hybrid (Fisher@attn + LRP@mlp) starting $(date -Iseconds) ===" | tee -a "$LOG_MASTER"

for f in "$BASE/config.json" "$SRC1/config.json" "$SRC2/config.json" "$LRP1" "$LRP2" "$F1" "$F2"; do
    [ -f "$f" ] || { echo "[!] missing: $f" | tee -a "$LOG_MASTER"; exit 1; }
done

# ─── Step 1: build hybrid score safetensors per source ──────────────────────
if [ ! -f "$HYB1" ]; then
    echo "[*] building hybrid score for jackrong-v2 ..." | tee -a "$LOG_MASTER"
    python3 "$WORKSPACE/scripts/build_hybrid_fisher_lrp.py" "$F1" "$LRP1" "$HYB1" 2>&1 | tee -a "$LOG_MASTER"
else
    echo "[*] hybrid jackrong-v2: SKIP" | tee -a "$LOG_MASTER"
fi
if [ ! -f "$HYB2" ]; then
    echo "[*] building hybrid score for crow-4b ..." | tee -a "$LOG_MASTER"
    python3 "$WORKSPACE/scripts/build_hybrid_fisher_lrp.py" "$F2" "$LRP2" "$HYB2" 2>&1 | tee -a "$LOG_MASTER"
else
    echo "[*] hybrid crow-4b: SKIP" | tee -a "$LOG_MASTER"
fi

# ─── Step 2: write lrp_config.yaml pointing at hybrid scores ────────────────
LRP_CFG="$OUT/lrp/m6_hybrid_config.yaml"
cat > "$LRP_CFG" <<EOF
merge_method: lrp

base_model:
  model: "$BASE"

parameters:
  density: 0.7

models:
  - model: "$SRC1"
    parameters:
      weight: 1.0
      lrp_scores: "$HYB1"

  - model: "$SRC2"
    parameters:
      weight: 1.0
      lrp_scores: "$HYB2"
EOF
echo "[*] wrote $LRP_CFG" | tee -a "$LOG_MASTER"

# ─── Step 3: mergekit-yaml merge ────────────────────────────────────────────
if [ ! -f "$M6_DIR/.merge_done" ] || ! find "$M6_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
    echo "[*] running mergekit-yaml merge ..." | tee -a "$LOG_MASTER"
    rm -rf "$M6_DIR"
    mkdir -p "$M6_DIR"
    "$MERGEKIT_PY" -u -m mergekit.scripts.run_yaml "$LRP_CFG" "$M6_DIR" --random-seed 42 \
        2>&1 | tee -a "$LOGS/4b_merge_m6.log"
    if find "$M6_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
        touch "$M6_DIR/.merge_done"
    else
        echo "[!] merge produced no safetensors — aborting" | tee -a "$LOG_MASTER"
        exit 1
    fi
else
    echo "[*] merge: SKIP (already done)" | tee -a "$LOG_MASTER"
fi

# ─── Step 4: convert + quantize ─────────────────────────────────────────────
if [ ! -f "$M6_Q6K" ]; then
    echo "[*] convert HF → F16 GGUF" | tee -a "$LOG_MASTER"
    python3 /opt/llama.cpp/convert_hf_to_gguf.py "$M6_DIR" --outtype f16 --outfile "$M6_F16" \
        2>&1 | tee -a "$LOGS/4b_convert_m6.log" | tail -10
    echo "[*] quantize Q6_K" | tee -a "$LOG_MASTER"
    "$LLAMA/llama-quantize" "$M6_F16" "$M6_Q6K" Q6_K 2>&1 | tee -a "$LOGS/4b_quantize_m6.log" | tail -5
    rm -f "$M6_F16"
else
    echo "[*] quantize: SKIP" | tee -a "$LOG_MASTER"
fi

# ─── Step 5: launch llama-server ────────────────────────────────────────────
pkill -f 'llama-server.*--port 8099' 2>/dev/null
sleep 2
nohup "$LLAMA/llama-server" \
    -m "$M6_Q6K" --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
    --reasoning-format deepseek --reasoning-budget 8192 \
    --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 \
    > "$LOGS/4b_server_m6.log" 2>&1 &
SERVER_PID=$!
disown
for i in $(seq 1 60); do
    curl -sf http://localhost:8099/health > /dev/null 2>&1 && break
    sleep 2
done
curl -sf http://localhost:8099/health > /dev/null 2>&1 || { echo "[!] server failed"; tail -30 "$LOGS/4b_server_m6.log"; exit 1; }
echo "    server ready (PID $SERVER_PID)" | tee -a "$LOG_MASTER"

# ─── Step 6: HE + MBPP eval ─────────────────────────────────────────────────
TOK_BASE="$BASE"
EVAL_DIR="$OUT/eval_results"
for task in mbpp humaneval; do
    echo "[*] $task eval $(date -Iseconds)" | tee -a "$LOG_MASTER"
    OUTPATH="$EVAL_DIR/${task}_${M6_NAME}"
    mkdir -p "$OUTPATH"
    /shared/dev/lightseek/.venv/bin/python -u -m lm_eval \
        --model local-completions \
        --model_args "model=${M6_NAME},base_url=http://localhost:8099/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=${TOK_BASE},max_gen_toks=2048" \
        --tasks "$task" \
        --gen_kwargs "temperature=0.0,top_p=1.0" \
        --batch_size 1 --use_cache "$OUTPATH/cache" \
        --log_samples --confirm_run_unsafe_code \
        --output_path "$OUTPATH" \
        2>&1 | tee -a "$LOGS/4b_${task}_m6.log" | tail -20
done

kill "$SERVER_PID" 2>/dev/null
sleep 2
echo "=== M6 done $(date -Iseconds) ===" | tee -a "$LOG_MASTER"
find "$EVAL_DIR" -path "*${M6_NAME}*" -name "results_*.json" 2>/dev/null
