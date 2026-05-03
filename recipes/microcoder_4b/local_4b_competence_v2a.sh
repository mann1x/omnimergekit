#!/bin/bash
# Competence-driven merge experiment v2a.
#
# Pipeline:
#   1. Smoke-test extractor on jackrong-v2 HE (5 samples, ~30s)
#   2. Full extraction: 4 sources × 2 tasks (HE+MBPP), 80 samples each
#   3. Combine per-source with raw success rates (no above-floor; all sources <= base on HE)
#   4. Remerge with omnimerge_v2 + obim,darex,emr,fisher (v1b recipe + competence)
#   5. Quantize Q6_K + eval HE+MBPP
set -uo pipefail

WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
HF=$WS/hf_models_4b
OUT=$WS/4b_phase1
LOGS=$WS/logs
LLAMA=/opt/llama.cpp/build/bin
PYBIN=/shared/dev/lightseek/.venv/bin/python
BASE=$HF/Qwen3.5-4B

NAME=${NAME:-v2a-competence}
COMP_DIR=$OUT/competence_maps
COMP_COMBINED=$COMP_DIR/combined
M_DIR=$OUT/merged/$NAME
M_F16=$OUT/gguf/$NAME-F16.gguf
M_Q6K=$OUT/gguf/$NAME-Q6_K.gguf
LOG_MASTER=$LOGS/4b_${NAME}.log
EVAL_DIR=$OUT/eval_results
mkdir -p $COMP_DIR $COMP_COMBINED $M_DIR $OUT/gguf $LOGS $EVAL_DIR
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1

echo "=== ${NAME} starting $(date -Iseconds) ===" | tee -a $LOG_MASTER

# Source list: (label, hf_subdir, he_samples_dir, mbpp_samples_dir, he_results_dir, mbpp_results_dir)
declare -A SRC_HF=(
    [jackrong-v2]=$HF/jackrong-v2
    [jackrong-python]=$HF/coder_eval/jackrong-python
    [continuum-forged]=$HF/coder_eval/continuum-code-forged
    [massivdash-ts]=$HF/coder_eval/massivdash-ts
)
declare -A EVAL_BN=(
    [jackrong-v2]=jackrong-v2
    [jackrong-python]=jackrong-python
    [continuum-forged]=continuum-code-forged
    [massivdash-ts]=massivdash-ts
)

# Phase 1: smoke test
SMOKE=$COMP_DIR/_smoke_jackrong-v2_he.safetensors
if [ ! -f "$SMOKE" ]; then
    echo "[smoke] jackrong-v2 HE × 5 samples ($(date +%H:%M:%S))" | tee -a $LOG_MASTER
    "$PYBIN" -u $WS/scripts/competence_extract.py \
        --model $HF/jackrong-v2 \
        --samples "$OUT/eval_results/humaneval_jackrong-v2/jackrong-v2/samples_humaneval_*.jsonl" \
        --task he \
        --output $SMOKE \
        --max-samples 5 --max-len 512 \
        2>&1 | tee -a $LOGS/4b_competence_smoke.log
    if [ ! -f "$SMOKE" ]; then
        echo "[!] smoke test failed — extractor did not produce output" | tee -a $LOG_MASTER
        exit 1
    fi
    echo "[smoke] OK ($(stat -c%s $SMOKE | numfmt --to=iec))" | tee -a $LOG_MASTER
fi

# Phase 2: full extraction
for src in jackrong-v2 jackrong-python continuum-forged; do
    bn=${EVAL_BN[$src]}
    for task in he mbpp; do
        OUT_F=$COMP_DIR/${src}__${task}.safetensors
        if [ -f "$OUT_F" ]; then
            echo "[extract] $src $task — already done, skip" | tee -a $LOG_MASTER
            continue
        fi
        # Map task → samples glob + lm_eval task name
        case $task in
            he)   GLOB="$OUT/eval_results/humaneval_${bn}/${bn}/samples_humaneval_*.jsonl" ;;
            mbpp) GLOB="$OUT/eval_results/mbpp_${bn}/${bn}/samples_mbpp_*.jsonl" ;;
        esac
        echo "[extract] $src $task ($(date +%H:%M:%S))" | tee -a $LOG_MASTER
        "$PYBIN" -u $WS/scripts/competence_extract.py \
            --model ${SRC_HF[$src]} \
            --samples "$GLOB" \
            --task $task \
            --output $OUT_F \
            --max-samples 50 --max-len 512 \
            2>&1 | tee -a $LOGS/4b_competence_${src}_${task}.log | grep -vE "^Writing:|^Loading|byte/s" | tail -30
        if [ ! -f "$OUT_F" ]; then
            echo "[!] extract failed: $src $task" | tee -a $LOG_MASTER
            exit 1
        fi
    done
done

# Phase 3: combine. Raw rates from results.json. Pass --raw-rate so all tasks contribute.
echo "[combine] ($(date +%H:%M:%S))" | tee -a $LOG_MASTER
MAP_ARGS=()
for src in jackrong-v2 jackrong-python continuum-forged; do
    bn=${EVAL_BN[$src]}
    HE_RES=$(ls $OUT/eval_results/humaneval_${bn}/${bn}/results_*.json | head -1)
    MB_RES=$(ls $OUT/eval_results/mbpp_${bn}/${bn}/results_*.json | head -1)
    MAP_ARGS+=(--map "${src}:humaneval:${HE_RES}:$COMP_DIR/${src}__he.safetensors")
    MAP_ARGS+=(--map "${src}:mbpp:${MB_RES}:$COMP_DIR/${src}__mbpp.safetensors")
done
"$PYBIN" -u $WS/scripts/competence_combine.py \
    "${MAP_ARGS[@]}" \
    --raw-rate \
    --signal weight_taylor \
    --output-dir $COMP_COMBINED \
    2>&1 | tee -a $LOGS/4b_competence_combine.log
ls -la $COMP_COMBINED | tee -a $LOG_MASTER
# Reclaim disk: per-task maps no longer needed after combine
echo "[combine] freeing per-task maps:" | tee -a $LOG_MASTER
du -sh $COMP_DIR/*.safetensors 2>/dev/null | tee -a $LOG_MASTER
rm -f $COMP_DIR/*__he.safetensors $COMP_DIR/*__mbpp.safetensors $COMP_DIR/_smoke_*.safetensors

# Phase 4: remerge — v1b recipe (best HE 60.98) + fisher path consuming competence
# 2-source: jackrong-v2 0.45 + continuum-forged 0.55, omnimerge_v2 d=0.53 darex-q=0.75
if [ ! -f "$M_DIR/.merge_done" ] || ! find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
    rm -rf $M_DIR; mkdir -p $M_DIR
    echo "[merge] ($(date +%H:%M:%S))" | tee -a $LOG_MASTER
    "$PYBIN" -u $WS/scripts/omnimergekit.py \
        --base $BASE \
        --source $HF/jackrong-v2 \
        --source $HF/coder_eval/continuum-code-forged \
        --output $M_DIR \
        --method omnimerge_v2 --v2-features obim,darex,emr,fisher \
        --weights 0.45,0.55 --density 0.53 --darex-q 0.75 \
        --fisher "$COMP_COMBINED/jackrong-v2.safetensors,$COMP_COMBINED/continuum-forged.safetensors" \
        --pr682-turbo \
        --m7-detector --m7-layer-aware \
        --skip-patterns "model.visual,mtp.layers" \
        --seed 42 --device cuda \
        2>&1 | tee -a $LOGS/4b_merge_${NAME}.log | tail -40
    if find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
        touch $M_DIR/.merge_done
    else
        echo "[!] $NAME merge produced no safetensors — aborting" | tee -a $LOG_MASTER; exit 1
    fi
fi

# Phase 5: convert + quantize
if [ ! -f "$M_Q6K" ]; then
    python3 /opt/llama.cpp/convert_hf_to_gguf.py $M_DIR --outtype f16 --outfile $M_F16 \
        2>&1 | tee -a $LOGS/4b_convert_${NAME}.log | tail -8
    [ -f "$M_F16" ] || { echo "[!] $NAME convert failed"; exit 1; }
    $LLAMA/llama-quantize $M_F16 $M_Q6K Q6_K 2>&1 | tee -a $LOGS/4b_quantize_${NAME}.log | tail -3
    [ -f "$M_Q6K" ] || { echo "[!] $NAME quantize failed"; exit 1; }
    rm -f $M_F16
fi

# Phase 6: serve + eval HE+MBPP
pkill -f 'llama-server.*--port 8099' 2>/dev/null
sleep 3
nohup $LLAMA/llama-server -m $M_Q6K --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
    --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 \
    > $LOGS/4b_server_${NAME}.log 2>&1 &
SERVER_PID=$!
disown
for i in $(seq 1 60); do curl -sf http://localhost:8099/health > /dev/null 2>&1 && break; sleep 2; done
curl -sf http://localhost:8099/health > /dev/null 2>&1 || { echo "[!] server failed"; tail -30 $LOGS/4b_server_${NAME}.log; exit 1; }
echo "[eval] server ready ($(date +%H:%M:%S))"

for task in mbpp humaneval; do
    OUTPATH=$EVAL_DIR/${task}_${NAME}; mkdir -p $OUTPATH
    "$PYBIN" -u -m lm_eval \
        --model local-completions \
        --model_args "model=${NAME},base_url=http://localhost:8099/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=${BASE},max_gen_toks=2048" \
        --tasks "$task" --gen_kwargs "temperature=0.0,top_p=1.0" \
        --batch_size 1 --use_cache "$OUTPATH/cache" --log_samples --confirm_run_unsafe_code \
        --output_path "$OUTPATH" 2>&1 | tee -a $LOGS/4b_${task}_${NAME}.log | tail -8
done

kill $SERVER_PID 2>/dev/null
echo "=== ${NAME} done $(date -Iseconds) ===" | tee -a $LOG_MASTER
