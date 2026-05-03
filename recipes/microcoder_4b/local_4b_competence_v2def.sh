#!/bin/bash
# v2d/v2e/v2f — sweep on top of v2c. Test 3 knobs to tune the HE/MBPP trade.
# All reuse the v2b differential combined competence maps.
set -uo pipefail

WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
HF=$WS/hf_models_4b
OUT=$WS/4b_phase1
LOGS=$WS/logs
LLAMA=/opt/llama.cpp/build/bin
PYBIN=/shared/dev/lightseek/.venv/bin/python
BASE=$HF/Qwen3.5-4B
COMP=$OUT/competence_maps_diff/combined

run_variant() {
    local NAME=$1; local FEATURES=$2; local DENSITY=$3; local DAREXQ=$4
    local M_DIR=$OUT/merged/$NAME
    local M_F16=$OUT/gguf/$NAME-F16.gguf
    local M_Q6K=$OUT/gguf/$NAME-Q6_K.gguf
    local LOG_MASTER=$LOGS/4b_${NAME}.log
    local EVAL_DIR=$OUT/eval_results
    mkdir -p $M_DIR $OUT/gguf $EVAL_DIR

    echo "" | tee -a $LOG_MASTER
    echo "=== $NAME starting $(date -Iseconds) features=$FEATURES density=$DENSITY darex-q=$DAREXQ ===" | tee -a $LOG_MASTER

    # Build optional --darex-q arg
    local EXTRA=""
    if [ "$DAREXQ" != "-" ]; then
        EXTRA="--darex-q $DAREXQ"
    fi

    # Phase 1: merge
    if [ ! -f "$M_DIR/.merge_done" ] || ! find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
        rm -rf $M_DIR; mkdir -p $M_DIR
        echo "[merge] ($(date +%H:%M:%S))" | tee -a $LOG_MASTER
        "$PYBIN" -u $WS/scripts/omnimergekit.py \
            --base $BASE \
            --source $HF/jackrong-v2 \
            --source $HF/coder_eval/continuum-code-forged \
            --output $M_DIR \
            --method omnimerge_v2 --v2-features "$FEATURES" \
            --weights 0.45,0.55 --density $DENSITY $EXTRA \
            --fisher "$COMP/jackrong-v2.safetensors,$COMP/continuum-forged.safetensors" \
            --pr682-turbo \
            --skip-patterns "model.visual,mtp.layers" \
            --seed 42 --device cuda \
            2>&1 | tee -a $LOGS/4b_merge_${NAME}.log | tail -20
        if find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
            touch $M_DIR/.merge_done
        else
            echo "[!] $NAME merge failed" | tee -a $LOG_MASTER; return 1
        fi
    fi

    # Phase 2: convert + quantize
    if [ ! -f "$M_Q6K" ]; then
        python3 /opt/llama.cpp/convert_hf_to_gguf.py $M_DIR --outtype f16 --outfile $M_F16 \
            2>&1 | tee -a $LOGS/4b_convert_${NAME}.log | tail -3
        [ -f "$M_F16" ] || { echo "[!] convert failed"; return 1; }
        $LLAMA/llama-quantize $M_F16 $M_Q6K Q6_K 2>&1 | tee -a $LOGS/4b_quantize_${NAME}.log | tail -2
        [ -f "$M_Q6K" ] || { echo "[!] quantize failed"; return 1; }
        rm -f $M_F16
    fi

    # Phase 3: serve + eval
    pkill -f 'llama-server.*--port 8099' 2>/dev/null
    sleep 3
    nohup $LLAMA/llama-server -m $M_Q6K --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
        --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 \
        > $LOGS/4b_server_${NAME}.log 2>&1 &
    local SERVER_PID=$!
    disown
    for i in $(seq 1 60); do curl -sf http://localhost:8099/health > /dev/null 2>&1 && break; sleep 2; done
    curl -sf http://localhost:8099/health > /dev/null 2>&1 || { echo "[!] server failed"; tail -10 $LOGS/4b_server_${NAME}.log; return 1; }
    echo "[eval] server ready ($(date +%H:%M:%S))"

    for task in mbpp humaneval; do
        local OUTPATH=$OUT/eval_results/${task}_${NAME}; mkdir -p $OUTPATH
        "$PYBIN" -u -m lm_eval \
            --model local-completions \
            --model_args "model=${NAME},base_url=http://localhost:8099/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=${BASE},max_gen_toks=2048" \
            --tasks "$task" --gen_kwargs "temperature=0.0,top_p=1.0" \
            --batch_size 1 --use_cache "$OUTPATH/cache" --log_samples --confirm_run_unsafe_code \
            --output_path "$OUTPATH" 2>&1 | tee -a $LOGS/4b_${task}_${NAME}.log | tail -5
    done

    kill $SERVER_PID 2>/dev/null
    sleep 3
    echo "=== $NAME done $(date -Iseconds) ===" | tee -a $LOG_MASTER
}

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1

# v2d: fisher-only + density 0.30 (sharper filter, more deltas suppressed)
run_variant "v2d-fisher-d030" "fisher" "0.30" "-" || true

# v2e: fisher + darex (less aggressive rescale q=0.85)
run_variant "v2e-fisher-darex" "fisher,darex" "0.53" "0.85" || true

# v2f: fisher + emr (no OBIM filter, but keep EMR election)
run_variant "v2f-fisher-emr" "fisher,emr" "0.53" "-" || true

echo ""
echo "=== ALL DONE $(date -Iseconds) ==="
echo "Final scores:"
for v in v2d-fisher-d030 v2e-fisher-darex v2f-fisher-emr; do
    HE=$(grep -E "pass@1" $LOGS/4b_humaneval_${v}.log 2>/dev/null | tail -1 | grep -oE "0\.[0-9]+" | head -1)
    MB=$(grep -E "pass_at_1" $LOGS/4b_mbpp_${v}.log 2>/dev/null | tail -1 | grep -oE "0\.[0-9]+" | head -1)
    printf "  %-20s HE=%s  MBPP=%s\n" "$v" "$HE" "$MB"
done
