#!/bin/bash
# M8 = M7v2 (layer-aware mergetime detector) + --pr682-turbo + magnitude-DARE.
# Combines all gating mechanisms (no importance signal). M2-recipe hyperparams.
set -uo pipefail

WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
HF=$WS/hf_models_4b
OUT=$WS/4b_phase1
LOGS=$WS/logs
LLAMA=/opt/llama.cpp/build/bin
PYBIN=/shared/dev/lightseek/.venv/bin/python

BASE=$HF/Qwen3.5-4B
SRC1=$HF/jackrong-v2
SRC2=$HF/crow-4b

NAME=m8
M_DIR=$OUT/merged/$NAME
M_F16=$OUT/gguf/$NAME-F16.gguf
M_Q6K=$OUT/gguf/$NAME-Q6_K.gguf
LOG_MASTER=$LOGS/4b_${NAME}.log
mkdir -p $M_DIR $OUT/gguf $LOGS
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1

echo "=== ${NAME} starting $(date -Iseconds) ===" | tee -a $LOG_MASTER

if [ ! -f "$M_DIR/.merge_done" ] || ! find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
    rm -rf $M_DIR; mkdir -p $M_DIR
    "$PYBIN" -u $WS/scripts/dare_ties_merge.py \
        --base $BASE --source $SRC1 --source $SRC2 --output $M_DIR \
        --method omnimerge_v2 --v2-features obim,darex,emr \
        --weights 0.55,0.45 --density 0.53 --darex-q 0.75 \
        --pr682-turbo \
        --m7-detector --m7-layer-aware \
        --skip-patterns "model.visual,mtp.layers" \
        --seed 42 --device cuda \
        2>&1 | tee -a $LOGS/4b_merge_${NAME}.log
    if find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
        touch $M_DIR/.merge_done
    else
        echo "[!] merge produced no safetensors — aborting" | tee -a $LOG_MASTER; exit 1
    fi
fi

if [ ! -f "$M_Q6K" ]; then
    python3 /opt/llama.cpp/convert_hf_to_gguf.py $M_DIR --outtype f16 --outfile $M_F16 2>&1 | tee -a $LOGS/4b_convert_${NAME}.log | tail -10
    $LLAMA/llama-quantize $M_F16 $M_Q6K Q6_K 2>&1 | tee -a $LOGS/4b_quantize_${NAME}.log | tail -5
    rm -f $M_F16
fi

pkill -f 'llama-server.*--port 8099' 2>/dev/null
sleep 2
nohup $LLAMA/llama-server -m $M_Q6K --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
    --reasoning-format deepseek --reasoning-budget 8192 \
    --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 \
    > $LOGS/4b_server_${NAME}.log 2>&1 &
SERVER_PID=$!
disown
for i in $(seq 1 60); do curl -sf http://localhost:8099/health > /dev/null 2>&1 && break; sleep 2; done
curl -sf http://localhost:8099/health > /dev/null 2>&1 || { echo "[!] server failed"; tail -30 $LOGS/4b_server_${NAME}.log; exit 1; }

EVAL_DIR=$OUT/eval_results
for task in mbpp humaneval; do
    OUTPATH=$EVAL_DIR/${task}_${NAME}
    mkdir -p $OUTPATH
    /shared/dev/lightseek/.venv/bin/python -u -m lm_eval \
        --model local-completions \
        --model_args "model=${NAME},base_url=http://localhost:8099/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=${BASE},max_gen_toks=2048" \
        --tasks "$task" --gen_kwargs "temperature=0.0,top_p=1.0" \
        --batch_size 1 --use_cache "$OUTPATH/cache" --log_samples --confirm_run_unsafe_code \
        --output_path "$OUTPATH" 2>&1 | tee -a $LOGS/4b_${task}_${NAME}.log | tail -15
done
kill $SERVER_PID 2>/dev/null
echo "=== ${NAME} done $(date -Iseconds) ===" | tee -a $LOG_MASTER
