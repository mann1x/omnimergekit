#!/bin/bash
# Parameterized MicroCoder variant runner — HE+MBPP only (skip LCB by default).
# Use as a building block for sweeps. Each variant is: merge + quant + HE + MBPP.
#
# Usage:
#   NAME=v1c \
#   SOURCES="jackrong-v2,coder_eval/continuum-code-forged" \
#   WEIGHTS="0.45,0.55" \
#   METHOD=dare_linear DENSITY=0.53 \
#   EXTRA_ARGS="--pr682-turbo --m7-detector --m7-layer-aware --darex-q 0.75 --v2-features obim,darex,emr" \
#   WITH_LCB=0 \
#   bash scripts/local_4b_microcoder_variant.sh
set -uo pipefail

WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
HF=$WS/hf_models_4b
OUT=$WS/4b_phase1
LOGS=$WS/logs
LLAMA=/opt/llama.cpp/build/bin
PYBIN=/shared/dev/lightseek/.venv/bin/python
BASE=$HF/Qwen3.5-4B

NAME=${NAME:?missing NAME}
SOURCES=${SOURCES:?missing SOURCES (csv of subdirs of hf_models_4b/)}
WEIGHTS=${WEIGHTS:?missing WEIGHTS (csv same len as SOURCES)}
METHOD=${METHOD:-omnimerge_v2}
DENSITY=${DENSITY:-0.53}
EXTRA_ARGS=${EXTRA_ARGS:-"--pr682-turbo --m7-detector --m7-layer-aware --darex-q 0.75 --v2-features obim,darex,emr"}
WITH_LCB=${WITH_LCB:-0}
SEED=${SEED:-42}

M_DIR=$OUT/merged/$NAME
M_F16=$OUT/gguf/$NAME-F16.gguf
M_Q6K=$OUT/gguf/$NAME-Q6_K.gguf
LOG_MASTER=$LOGS/4b_${NAME}.log
EVAL_DIR=$OUT/eval_results
mkdir -p $M_DIR $OUT/gguf $LOGS $EVAL_DIR
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1

# Wait for any blocking pipelines (v1, v1b) + llama-server on 8099 to vacate.
# DON'T match sweep.sh / variant.sh themselves — sweep is the parent we're spawned from.
echo "[$NAME] waiting for v1/v1b pipelines + port 8099"
while pgrep -f "local_4b_microcoder\.sh\|local_4b_microcoder_v1b\.sh" >/dev/null 2>&1; do sleep 15; done
while pgrep -f "llama-server.*--port 8099" >/dev/null 2>&1; do sleep 10; done
sleep 5
echo "=== ${NAME} starting $(date -Iseconds) ===" | tee -a $LOG_MASTER
echo "  method=$METHOD density=$DENSITY weights=$WEIGHTS sources=$SOURCES" | tee -a $LOG_MASTER

# Build --source args
SRC_ARGS=()
IFS=',' read -ra SRC_LIST <<< "$SOURCES"
for s in "${SRC_LIST[@]}"; do
    SRC_ARGS+=(--source "$HF/$s")
done

if [ ! -f "$M_DIR/.merge_done" ] || ! find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
    rm -rf $M_DIR; mkdir -p $M_DIR
    "$PYBIN" -u $WS/scripts/dare_ties_merge.py \
        --base $BASE \
        "${SRC_ARGS[@]}" \
        --output $M_DIR \
        --method $METHOD \
        --weights $WEIGHTS --density $DENSITY \
        $EXTRA_ARGS \
        --skip-patterns "model.visual,mtp.layers" \
        --seed $SEED --device cuda \
        2>&1 | tee -a $LOGS/4b_merge_${NAME}.log
    if find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
        touch $M_DIR/.merge_done
    else
        echo "[!] $NAME merge produced no safetensors — aborting" | tee -a $LOG_MASTER; exit 1
    fi
fi

if [ ! -f "$M_Q6K" ]; then
    python3 /opt/llama.cpp/convert_hf_to_gguf.py $M_DIR --outtype f16 --outfile $M_F16 \
        2>&1 | tee -a $LOGS/4b_convert_${NAME}.log | tail -8
    [ -f "$M_F16" ] || { echo "[!] $NAME convert failed"; exit 1; }
    $LLAMA/llama-quantize $M_F16 $M_Q6K Q6_K 2>&1 | tee -a $LOGS/4b_quantize_${NAME}.log | tail -3
    [ -f "$M_Q6K" ] || { echo "[!] $NAME quantize failed"; exit 1; }
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
curl -sf http://localhost:8099/health > /dev/null 2>&1 || { echo "[!] $NAME server failed"; tail -30 $LOGS/4b_server_${NAME}.log; exit 1; }
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

if [ "$WITH_LCB" = "1" ]; then
    LP=$EVAL_DIR/lcb_${NAME}; mkdir -p $LP
    "$PYBIN" -u $WS/scripts/lcb_llama_server.py \
        --name "$NAME" --base-url http://localhost:8099 \
        --limit 30 --difficulty medium --min-date 2024-10-01 \
        --max-tokens 8192 \
        --output "$LP/lcb_results.json" --samples "$LP/lcb_samples.jsonl" \
        2>&1 | tee -a $LOGS/4b_lcb_${NAME}.log | grep -E "PASS|FAIL|pass@1|Error" | tail -35
fi

kill $SERVER_PID 2>/dev/null
echo "=== ${NAME} done $(date -Iseconds) ===" | tee -a $LOG_MASTER
