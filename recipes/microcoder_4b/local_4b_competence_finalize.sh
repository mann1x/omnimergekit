#!/bin/bash
# Finalize: LCB-30 on v2e (best 2-source) + build 3-source v2g + eval all three.
set -uo pipefail

WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
HF=$WS/hf_models_4b
OUT=$WS/4b_phase1
LOGS=$WS/logs
LLAMA=/opt/llama.cpp/build/bin
PYBIN=/shared/dev/lightseek/.venv/bin/python
BASE=$HF/Qwen3.5-4B
EVAL_DIR=$OUT/eval_results
COMP3=$OUT/competence_maps_3way
COMP3_COMBINED=$COMP3/combined
DIFFSETS=$COMP3/_diffsets.json
mkdir -p $COMP3 $COMP3_COMBINED
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1

# Helper: serve and run a single task or LCB
serve_and_eval() {
    local NAME=$1; local GGUF=$2; local DO_LCB=$3
    pkill -f 'llama-server.*--port 8099' 2>/dev/null
    sleep 3
    nohup $LLAMA/llama-server -m $GGUF --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
        --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 \
        > $LOGS/4b_server_${NAME}.log 2>&1 &
    local SERVER_PID=$!
    disown
    for i in $(seq 1 60); do curl -sf http://localhost:8099/health > /dev/null 2>&1 && break; sleep 2; done
    curl -sf http://localhost:8099/health > /dev/null 2>&1 || { echo "[!] $NAME server failed"; return 1; }

    if [ "$DO_LCB" = "1" ]; then
        local LP=$EVAL_DIR/lcb_${NAME}; mkdir -p $LP
        "$PYBIN" -u $WS/scripts/lcb_llama_server.py \
            --name "$NAME" --base-url http://localhost:8099 \
            --limit 30 --difficulty medium --min-date 2024-10-01 \
            --max-tokens 8192 \
            --output "$LP/lcb_results.json" --samples "$LP/lcb_samples.jsonl" \
            2>&1 | tee -a $LOGS/4b_lcb_${NAME}.log | grep -E "PASS|FAIL|pass@1|Error" | tail -10
    fi

    kill $SERVER_PID 2>/dev/null
    sleep 3
}

# ============ Phase 1: LCB-30 on v2e ============
echo ""
echo "=== Phase 1: LCB-30 on v2e-fisher-darex $(date +%H:%M:%S) ==="
V2E_GGUF=$OUT/gguf/v2e-fisher-darex-Q6_K.gguf
[ -f "$V2E_GGUF" ] || { echo "[!] v2e GGUF missing"; exit 1; }
serve_and_eval "v2e-fisher-darex" "$V2E_GGUF" 1

# ============ Phase 2: extract 3-way differential maps ============
declare -A SRC_HF=(
    [jackrong-v2]=$HF/jackrong-v2
    [continuum-forged]=$HF/coder_eval/continuum-code-forged
    [jackrong-python]=$HF/coder_eval/jackrong-python
)
declare -A EVAL_BN=(
    [jackrong-v2]=jackrong-v2
    [continuum-forged]=continuum-code-forged
    [jackrong-python]=jackrong-python
)

for src in jackrong-v2 continuum-forged jackrong-python; do
    bn=${EVAL_BN[$src]}
    for task in he mbpp; do
        OUT_F=$COMP3/${src}__${task}.safetensors
        if [ -f "$OUT_F" ]; then
            echo "[extract3] $src $task — already done, skip"
            continue
        fi
        case $task in
            he)   GLOB="$OUT/eval_results/humaneval_${bn}/${bn}/samples_humaneval_*.jsonl" ;;
            mbpp) GLOB="$OUT/eval_results/mbpp_${bn}/${bn}/samples_mbpp_*.jsonl" ;;
        esac
        DOC_IDS=$("$PYBIN" -c "import json; d=json.load(open('$DIFFSETS')); print(','.join(d['${src}__${task}']))")
        N_DOCS=$(echo "$DOC_IDS" | tr ',' '\n' | grep -c .)
        echo "[extract3] $src $task — $N_DOCS docs ($(date +%H:%M:%S))"
        "$PYBIN" -u $WS/scripts/competence_extract.py \
            --model ${SRC_HF[$src]} \
            --samples "$GLOB" \
            --task $task \
            --keep-doc-ids "$DOC_IDS" \
            --output $OUT_F \
            --max-samples 200 --max-len 512 \
            2>&1 | tee -a $LOGS/4b_competence_3way_${src}_${task}.log | grep -vE "^Writing:|^Loading|byte/s" | tail -10
        if [ ! -f "$OUT_F" ]; then
            echo "[!] extract failed: $src $task"; exit 1
        fi
    done
done

# ============ Phase 3: combine 3-way ============
echo ""
echo "=== Phase 3: combine $(date +%H:%M:%S) ==="
MAP_ARGS=()
for src in jackrong-v2 continuum-forged jackrong-python; do
    bn=${EVAL_BN[$src]}
    HE_RES=$(ls $OUT/eval_results/humaneval_${bn}/${bn}/results_*.json | head -1)
    MB_RES=$(ls $OUT/eval_results/mbpp_${bn}/${bn}/results_*.json | head -1)
    MAP_ARGS+=(--map "${src}:humaneval:${HE_RES}:$COMP3/${src}__he.safetensors")
    MAP_ARGS+=(--map "${src}:mbpp:${MB_RES}:$COMP3/${src}__mbpp.safetensors")
done
"$PYBIN" -u $WS/scripts/competence_combine.py \
    "${MAP_ARGS[@]}" \
    --raw-rate \
    --signal weight_taylor \
    --output-dir $COMP3_COMBINED \
    2>&1 | tee -a $LOGS/4b_competence_3way_combine.log
ls -la $COMP3_COMBINED
rm -f $COMP3/*__he.safetensors $COMP3/*__mbpp.safetensors

# ============ Phase 4: merge v2g (3-source) ============
NAME=v2g-3src-fisher-darex
M_DIR=$OUT/merged/$NAME
M_F16=$OUT/gguf/$NAME-F16.gguf
M_Q6K=$OUT/gguf/$NAME-Q6_K.gguf
mkdir -p $M_DIR

echo ""
echo "=== Phase 4: merge $NAME $(date +%H:%M:%S) ==="
if [ ! -f "$M_DIR/.merge_done" ] || ! find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
    rm -rf $M_DIR; mkdir -p $M_DIR
    "$PYBIN" -u $WS/scripts/omnimergekit.py \
        --base $BASE \
        --source $HF/jackrong-v2 \
        --source $HF/coder_eval/continuum-code-forged \
        --source $HF/coder_eval/jackrong-python \
        --output $M_DIR \
        --method omnimerge_v2 --v2-features fisher,darex \
        --weights 0.35,0.40,0.25 --density 0.53 --darex-q 0.85 \
        --fisher "$COMP3_COMBINED/jackrong-v2.safetensors,$COMP3_COMBINED/continuum-forged.safetensors,$COMP3_COMBINED/jackrong-python.safetensors" \
        --pr682-turbo \
        --skip-patterns "model.visual,mtp.layers" \
        --seed 42 --device cuda \
        2>&1 | tee -a $LOGS/4b_merge_${NAME}.log | tail -25
    if find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
        touch $M_DIR/.merge_done
    else
        echo "[!] $NAME merge failed"; exit 1
    fi
fi

# ============ Phase 5: convert + quantize ============
if [ ! -f "$M_Q6K" ]; then
    python3 /opt/llama.cpp/convert_hf_to_gguf.py $M_DIR --outtype f16 --outfile $M_F16 \
        2>&1 | tee -a $LOGS/4b_convert_${NAME}.log | tail -5
    [ -f "$M_F16" ] || { echo "[!] convert failed"; exit 1; }
    $LLAMA/llama-quantize $M_F16 $M_Q6K Q6_K 2>&1 | tee -a $LOGS/4b_quantize_${NAME}.log | tail -3
    [ -f "$M_Q6K" ] || { echo "[!] quantize failed"; exit 1; }
    rm -f $M_F16
fi

# ============ Phase 6: eval v2g HE+MBPP+LCB ============
echo ""
echo "=== Phase 6: eval $NAME $(date +%H:%M:%S) ==="
pkill -f 'llama-server.*--port 8099' 2>/dev/null
sleep 3
nohup $LLAMA/llama-server -m $M_Q6K --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
    --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 \
    > $LOGS/4b_server_${NAME}.log 2>&1 &
SERVER_PID=$!
disown
for i in $(seq 1 60); do curl -sf http://localhost:8099/health > /dev/null 2>&1 && break; sleep 2; done
curl -sf http://localhost:8099/health > /dev/null 2>&1 || { echo "[!] $NAME server failed"; tail -20 $LOGS/4b_server_${NAME}.log; exit 1; }
echo "[eval] server ready"

for task in mbpp humaneval; do
    OUTPATH=$EVAL_DIR/${task}_${NAME}; mkdir -p $OUTPATH
    "$PYBIN" -u -m lm_eval \
        --model local-completions \
        --model_args "model=${NAME},base_url=http://localhost:8099/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=${BASE},max_gen_toks=2048" \
        --tasks "$task" --gen_kwargs "temperature=0.0,top_p=1.0" \
        --batch_size 1 --use_cache "$OUTPATH/cache" --log_samples --confirm_run_unsafe_code \
        --output_path "$OUTPATH" 2>&1 | tee -a $LOGS/4b_${task}_${NAME}.log | tail -5
done

# LCB on v2g
LP=$EVAL_DIR/lcb_${NAME}; mkdir -p $LP
"$PYBIN" -u $WS/scripts/lcb_llama_server.py \
    --name "$NAME" --base-url http://localhost:8099 \
    --limit 30 --difficulty medium --min-date 2024-10-01 \
    --max-tokens 8192 \
    --output "$LP/lcb_results.json" --samples "$LP/lcb_samples.jsonl" \
    2>&1 | tee -a $LOGS/4b_lcb_${NAME}.log | grep -E "PASS|FAIL|pass@1|Error" | tail -10

kill $SERVER_PID 2>/dev/null
echo ""
echo "=== ALL DONE $(date -Iseconds) ==="

# Summary
echo ""
echo "=== Final summary ==="
for v in v1b v2e-fisher-darex v2g-3src-fisher-darex; do
    HE=$(grep -E "pass@1" $LOGS/4b_humaneval_${v}.log 2>/dev/null | tail -1 | grep -oE "0\.[0-9]+" | head -1)
    MB=$(grep -E "pass_at_1" $LOGS/4b_mbpp_${v}.log 2>/dev/null | tail -1 | grep -oE "0\.[0-9]+" | head -1)
    LC=$(python3 -c "import json,os; p='$EVAL_DIR/lcb_$v/lcb_results.json'; print(round(json.load(open(p))['pass_at_1']*100,2)) if os.path.exists(p) else print('-')" 2>/dev/null)
    printf "  %-25s HE=%s  MBPP=%s  LCB=%s\n" "$v" "${HE:-N/A}" "${MB:-N/A}" "${LC:-N/A}"
done
