#!/bin/bash
# microcoder-v1b: 2-source dare_ties test
#   jackrong-v2 (reasoning carrier)        weight 0.45
#   continuum-code-forged (MBPP carrier)   weight 0.55
#
# Hypothesis: dropping jackrong-python (the 3rd voter) lets continuum-forged's
# MBPP-discriminative deltas survive TIES sign-consensus. v1 saw MBPP drop to
# 45.40 (below floor); if v1b lands MBPP > 50, the 3-source TIES filter was
# the culprit.
#
# Same recipe otherwise: omnimerge_v2 + obim,darex,emr + pr682-turbo + m7
# layer-aware, density 0.53, darex-q 0.75, magnitude DARE, seed 42.
set -uo pipefail

WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
HF=$WS/hf_models_4b
OUT=$WS/4b_phase1
LOGS=$WS/logs
LLAMA=/opt/llama.cpp/build/bin
PYBIN=/shared/dev/lightseek/.venv/bin/python

BASE=$HF/Qwen3.5-4B
SRC1=$HF/jackrong-v2
SRC2=$HF/coder_eval/continuum-code-forged

NAME=microcoder-v1b
M_DIR=$OUT/merged/$NAME
M_F16=$OUT/gguf/$NAME-F16.gguf
M_Q6K=$OUT/gguf/$NAME-Q6_K.gguf
LOG_MASTER=$LOGS/4b_${NAME}.log
EVAL_DIR=$OUT/eval_results
mkdir -p $M_DIR $OUT/gguf $LOGS $EVAL_DIR
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1

# Wait for v1 pipeline + llama-server on 8099 to vacate
echo "[$NAME] waiting for v1 microcoder pipeline + port 8099"
while pgrep -f "local_4b_microcoder.sh\b" >/dev/null 2>&1; do sleep 30; done
while pgrep -f "llama-server.*--port 8099" >/dev/null 2>&1; do sleep 15; done
sleep 5
echo "=== ${NAME} starting $(date -Iseconds) ===" | tee -a $LOG_MASTER

if [ ! -f "$M_DIR/.merge_done" ] || ! find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
    rm -rf $M_DIR; mkdir -p $M_DIR
    "$PYBIN" -u $WS/scripts/dare_ties_merge.py \
        --base $BASE \
        --source $SRC1 --source $SRC2 \
        --output $M_DIR \
        --method omnimerge_v2 --v2-features obim,darex,emr \
        --weights 0.45,0.55 --density 0.53 --darex-q 0.75 \
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
    python3 /opt/llama.cpp/convert_hf_to_gguf.py $M_DIR --outtype f16 --outfile $M_F16 \
        2>&1 | tee -a $LOGS/4b_convert_${NAME}.log | tail -10
    [ -f "$M_F16" ] || { echo "[!] convert failed"; exit 1; }
    $LLAMA/llama-quantize $M_F16 $M_Q6K Q6_K 2>&1 | tee -a $LOGS/4b_quantize_${NAME}.log | tail -5
    [ -f "$M_Q6K" ] || { echo "[!] quantize failed"; exit 1; }
    rm -f $M_F16
fi

pkill -f 'llama-server.*--port 8099' 2>/dev/null
sleep 3
nohup $LLAMA/llama-server -m $M_Q6K --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
    --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 \
    > $LOGS/4b_server_${NAME}.log 2>&1 &
SERVER_PID=$!
disown
for i in $(seq 1 60); do curl -sf http://localhost:8099/health > /dev/null 2>&1 && break; sleep 2; done
curl -sf http://localhost:8099/health > /dev/null 2>&1 || { echo "[!] server failed"; tail -30 $LOGS/4b_server_${NAME}.log; exit 1; }
echo "  server ready"

for task in mbpp humaneval; do
    OUTPATH=$EVAL_DIR/${task}_${NAME}; mkdir -p $OUTPATH
    "$PYBIN" -u -m lm_eval \
        --model local-completions \
        --model_args "model=${NAME},base_url=http://localhost:8099/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=${BASE},max_gen_toks=2048" \
        --tasks "$task" --gen_kwargs "temperature=0.0,top_p=1.0" \
        --batch_size 1 --use_cache "$OUTPATH/cache" --log_samples --confirm_run_unsafe_code \
        --output_path "$OUTPATH" 2>&1 | tee -a $LOGS/4b_${task}_${NAME}.log | tail -8
done

LP=$EVAL_DIR/lcb_${NAME}; mkdir -p $LP
"$PYBIN" -u $WS/scripts/lcb_llama_server.py \
    --name "$NAME" --base-url http://localhost:8099 \
    --limit 30 --difficulty medium --min-date 2024-10-01 \
    --max-tokens 8192 \
    --output "$LP/lcb_results.json" --samples "$LP/lcb_samples.jsonl" \
    2>&1 | tee -a $LOGS/4b_lcb_${NAME}.log | grep -E "PASS|FAIL|pass@1|Error" | tail -35

kill $SERVER_PID 2>/dev/null
echo "=== ${NAME} done $(date -Iseconds) ===" | tee -a $LOG_MASTER
