#!/bin/bash
# v2i — Task-arithmetic merge: jackrong-v2 as merge_base, deltas vs Qwen3.5-4B.
#
# Hypothesis: v2g's AIME washout came from averaging jv's reasoning-direction delta
# with cf/jp (which have ZERO contribution in that subspace). v2h's fisher-blend
# couldn't fix it because 8 docs is too sparse a signal.
#
# This recipe takes a different angle:
#   - merge_base  = jackrong-v2     (pass through unchanged → reasoning preserved)
#   - task_base   = Qwen3.5-4B      (deltas computed FROM the shared ancestor)
#   - sources     = cf + jp         (their pure code/python task vectors)
#
#   merged = jackrong-v2 + w_cf · DARE(cf − Qwen3.5-4B) + w_jp · DARE(jp − Qwen3.5-4B)
#
# jackrong-v2 itself is NOT a source — it's the foundation. cf and jp only inject
# their *task-arithmetic* deltas. Fisher importance still consumed from their
# combined competence maps to keep the per-element sparsity behavior.
#
# Requires omnimergekit.py with --task-base support (added 2026-05-03).
set -uo pipefail

WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
HF=$WS/hf_models_4b
OUT=$WS/4b_phase1
LOGS=$WS/logs
LLAMA=/opt/llama.cpp/build/bin
PYBIN=/shared/dev/lightseek/.venv/bin/python
BASE=$HF/Qwen3.5-4B
JV=$HF/jackrong-v2
EVAL_DIR=$OUT/eval_results
COMP3=$OUT/competence_maps_3way
COMP3_COMBINED=$COMP3/combined
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1

NAME=v2i-jv-base-task-arith
M_DIR=$OUT/merged/$NAME
M_F16=$OUT/gguf/$NAME-F16.gguf
M_Q6K=$OUT/gguf/$NAME-Q6_K.gguf
LOG_MASTER=$LOGS/4b_${NAME}.log
mkdir -p $M_DIR $OUT/gguf

echo "=== $NAME starting $(date -Iseconds) ===" | tee $LOG_MASTER

# ============ Phase 1: merge with --task-base ============
if [ ! -f "$M_DIR/.merge_done" ] || ! find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
    rm -rf $M_DIR; mkdir -p $M_DIR
    echo "[merge] task-arithmetic: base=jackrong-v2, task_base=Qwen3.5-4B, sources=cf+jp ($(date +%H:%M:%S))" | tee -a $LOG_MASTER
    "$PYBIN" -u $WS/scripts/omnimergekit.py \
        --base $JV \
        --task-base $BASE \
        --source $HF/coder_eval/continuum-code-forged \
        --source $HF/coder_eval/jackrong-python \
        --output $M_DIR \
        --method omnimerge_v2 --v2-features fisher,darex \
        --weights 0.55,0.45 --density 0.53 --darex-q 0.85 \
        --fisher "$COMP3_COMBINED/continuum-forged.safetensors,$COMP3_COMBINED/jackrong-python.safetensors" \
        --pr682-turbo \
        --skip-patterns "model.visual,mtp.layers" \
        --seed 42 --device cuda \
        2>&1 | tee -a $LOGS/4b_merge_${NAME}.log | tail -30
    if find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
        touch $M_DIR/.merge_done
    else
        echo "[!] $NAME merge failed" | tee -a $LOG_MASTER; exit 1
    fi
fi

# ============ Phase 2: convert + quantize ============
if [ ! -f "$M_Q6K" ]; then
    echo "[convert+quant] $(date +%H:%M:%S)" | tee -a $LOG_MASTER
    "$PYBIN" /opt/llama.cpp/convert_hf_to_gguf.py $M_DIR --outtype f16 --outfile $M_F16 \
        2>&1 | tee -a $LOGS/4b_convert_${NAME}.log | tail -5
    [ -f "$M_F16" ] || { echo "[!] convert failed"; exit 1; }
    $LLAMA/llama-quantize $M_F16 $M_Q6K Q6_K 2>&1 | tee -a $LOGS/4b_quantize_${NAME}.log | tail -3
    [ -f "$M_Q6K" ] || { echo "[!] quantize failed"; exit 1; }
    rm -f $M_F16
fi

# ============ Phase 3: eval HE + MBPP + LCB ============
echo "" | tee -a $LOG_MASTER
echo "=== Phase 3: eval $NAME HE/MBPP/LCB $(date +%H:%M:%S) ===" | tee -a $LOG_MASTER
pkill -f 'llama-server.*--port 8099' 2>/dev/null
sleep 3
nohup $LLAMA/llama-server -m $M_Q6K --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
    --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 \
    > $LOGS/4b_server_${NAME}.log 2>&1 &
SERVER_PID=$!
disown
for i in $(seq 1 60); do curl -sf http://localhost:8099/health > /dev/null 2>&1 && break; sleep 2; done
curl -sf http://localhost:8099/health > /dev/null 2>&1 || { echo "[!] $NAME server failed"; tail -20 $LOGS/4b_server_${NAME}.log; exit 1; }

for task in mbpp humaneval; do
    OUTPATH=$EVAL_DIR/${task}_${NAME}; mkdir -p $OUTPATH
    "$PYBIN" -u -m lm_eval \
        --model local-completions \
        --model_args "model=${NAME},base_url=http://localhost:8099/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=${BASE},max_gen_toks=2048" \
        --tasks "$task" --gen_kwargs "temperature=0.0,top_p=1.0" \
        --batch_size 1 --use_cache "$OUTPATH/cache" --log_samples --confirm_run_unsafe_code \
        --output_path "$OUTPATH" 2>&1 | tee -a $LOGS/4b_${task}_${NAME}.log | tail -5
done

# LCB
LP=$EVAL_DIR/lcb_${NAME}; mkdir -p $LP
"$PYBIN" -u $WS/scripts/lcb_llama_server.py \
    --name "$NAME" --base-url http://localhost:8099 \
    --limit 30 --difficulty medium --min-date 2024-10-01 \
    --max-tokens 8192 \
    --output "$LP/lcb_results.json" --samples "$LP/lcb_samples.jsonl" \
    2>&1 | tee -a $LOGS/4b_lcb_${NAME}.log | grep -E "PASS|FAIL|pass@1|Error" | tail -10

kill $SERVER_PID 2>/dev/null
sleep 3

# ============ Phase 4: extended eval ============
echo "" | tee -a $LOG_MASTER
echo "=== Phase 4: extended eval $NAME $(date +%H:%M:%S) ===" | tee -a $LOG_MASTER
pkill -f 'llama-server.*--port 8099' 2>/dev/null
sleep 3
nohup $LLAMA/llama-server -m $M_Q6K --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
    --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 \
    > $LOGS/4b_ext_server_${NAME}.log 2>&1 &
SERVER_PID=$!
disown
for i in $(seq 1 60); do curl -sf http://localhost:8099/health > /dev/null 2>&1 && break; sleep 2; done
curl -sf http://localhost:8099/health > /dev/null 2>&1 || { echo "[!] $NAME ext server failed"; exit 1; }

for entry in "gsm8k:100:512" "mmlu_pro:200:1024" "aime:30:8192" "humaneval_plus:164:2048"; do
    TASK=${entry%%:*}
    rest=${entry#*:}
    LIMIT=${rest%%:*}
    MAXTOK=${rest##*:}
    OUTPATH=$EVAL_DIR/ext_${TASK}_${NAME}
    mkdir -p $OUTPATH
    echo "[ext-eval] $NAME $TASK limit=$LIMIT maxtok=$MAXTOK ($(date +%H:%M:%S))" | tee -a $LOG_MASTER
    "$PYBIN" -u -m lm_eval \
        --model local-completions \
        --model_args "model=${NAME},base_url=http://localhost:8099/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=${BASE},max_length=32768,max_gen_toks=${MAXTOK}" \
        --tasks "$TASK" --limit "$LIMIT" --gen_kwargs "temperature=0.0,top_p=1.0,max_gen_toks=${MAXTOK}" \
        --batch_size 1 --use_cache "$OUTPATH/cache" --log_samples --confirm_run_unsafe_code \
        --output_path "$OUTPATH" 2>&1 | tee -a $LOGS/4b_ext_${TASK}_${NAME}.log | tail -8 | tee -a $LOG_MASTER
done

kill $SERVER_PID 2>/dev/null
echo "" | tee -a $LOG_MASTER
echo "=== ALL DONE $(date -Iseconds) ===" | tee -a $LOG_MASTER

# Summary
echo "" | tee -a $LOG_MASTER
echo "=== $NAME scores ===" | tee -a $LOG_MASTER
for task_dir in mbpp humaneval lcb ext_gsm8k ext_mmlu_pro ext_aime ext_humaneval_plus; do
    case $task_dir in
        lcb) F=$EVAL_DIR/lcb_${NAME}/lcb_results.json; KEY="pass_at_1" ;;
        *)   F=$(ls $EVAL_DIR/${task_dir}_${NAME}/${NAME}/results_*.json 2>/dev/null | head -1); KEY="" ;;
    esac
    [ -z "$F" ] || [ ! -f "$F" ] && { echo "  $task_dir: -" | tee -a $LOG_MASTER; continue; }
    if [ -n "$KEY" ]; then
        python3 -c "import json; d=json.load(open('$F')); print(f'  $task_dir: {d[\"$KEY\"]*100:.2f}')" | tee -a $LOG_MASTER
    else
        python3 -c "
import json
d = json.load(open('$F'))['results']
for k, vals in d.items():
    for m, v in vals.items():
        if 'pass' in m.lower() and 'stderr' not in m and isinstance(v, float):
            print(f'  $task_dir: {v*100:.2f}'); exit()
        if 'exact_match' in m and 'stderr' not in m and isinstance(v, float):
            print(f'  $task_dir: {v*100:.2f}'); exit()
" | tee -a $LOG_MASTER
    fi
done
