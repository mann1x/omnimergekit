#!/bin/bash
# Extended reasoning eval on MicroCoder candidate + sources.
# Tasks: GSM8K-100, MMLU-Pro-200, AIME-30, HumanEval-Plus-164.
# (HumanEval-TS not in lm_eval; humaneval_plus substituted as harder code-gen probe.)
#
# Models: 3 sources (re-quantized from HF) + best 2-source variant (v2e) + final 3-source (v2g).
# Gated on v2g existence — waits if finalize pipeline still running.
set -uo pipefail

WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
HF=$WS/hf_models_4b
OUT=$WS/4b_phase1
LOGS=$WS/logs
LLAMA=/opt/llama.cpp/build/bin
PYBIN=/shared/dev/lightseek/.venv/bin/python
BASE=$HF/Qwen3.5-4B
EVAL_DIR=$OUT/eval_results
EXTQ=$OUT/gguf/_extended_sources
mkdir -p $EXTQ
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1

LOG_MASTER=$LOGS/4b_extended.log
echo "=== extended eval starting $(date -Iseconds) ===" | tee -a $LOG_MASTER

# Wait for finalize (v2g GGUF) to land
V2G_GGUF=$OUT/gguf/v2g-3src-fisher-darex-Q6_K.gguf
echo "[wait] waiting for $V2G_GGUF and finalize pipeline to finish" | tee -a $LOG_MASTER
while [ ! -f "$V2G_GGUF" ] || pgrep -f "local_4b_competence_finalize\.sh" >/dev/null 2>&1; do
    sleep 60
done
sleep 10
echo "[wait] v2g ready, starting extended eval" | tee -a $LOG_MASTER

# Re-quantize sources (skip if GGUF already exists)
quant_source() {
    local NAME=$1; local SRC_DIR=$2
    local F16=$EXTQ/${NAME}-F16.gguf
    local Q6K=$EXTQ/${NAME}-Q6_K.gguf
    if [ -f "$Q6K" ]; then echo "[quant] $NAME — already done"; return; fi
    echo "[quant] $NAME ($(date +%H:%M:%S))" | tee -a $LOG_MASTER
    "$PYBIN" /opt/llama.cpp/convert_hf_to_gguf.py $SRC_DIR --outtype f16 --outfile $F16 \
        2>&1 | tee -a $LOGS/4b_ext_convert_${NAME}.log | tail -5
    [ -f "$F16" ] || { echo "[!] convert $NAME failed" | tee -a $LOG_MASTER; exit 1; }
    $LLAMA/llama-quantize $F16 $Q6K Q6_K 2>&1 | tee -a $LOGS/4b_ext_quantize_${NAME}.log | tail -3
    [ -f "$Q6K" ] || { echo "[!] quantize $NAME failed" | tee -a $LOG_MASTER; exit 1; }
    rm -f $F16
}

quant_source "src_jackrong-v2"      $HF/jackrong-v2
quant_source "src_continuum-forged" $HF/coder_eval/continuum-code-forged
quant_source "src_jackrong-python"  $HF/coder_eval/jackrong-python

# Run extended eval on a model
ext_eval() {
    local NAME=$1; local GGUF=$2
    echo "" | tee -a $LOG_MASTER
    echo "=== eval $NAME $(date +%H:%M:%S) ===" | tee -a $LOG_MASTER
    pkill -f 'llama-server.*--port 8099' 2>/dev/null
    sleep 3
    nohup $LLAMA/llama-server -m $GGUF --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
        --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 \
        > $LOGS/4b_ext_server_${NAME}.log 2>&1 &
    local SERVER_PID=$!
    disown
    for i in $(seq 1 60); do curl -sf http://localhost:8099/health > /dev/null 2>&1 && break; sleep 2; done
    curl -sf http://localhost:8099/health > /dev/null 2>&1 || { echo "[!] $NAME server failed"; tail -20 $LOGS/4b_ext_server_${NAME}.log; return 1; }

    for entry in "gsm8k:100:512" "mmlu_pro:200:1024" "aime:30:8192" "humaneval_plus:164:2048"; do
        local TASK=${entry%%:*}
        local rest=${entry#*:}
        local LIMIT=${rest%%:*}
        local MAXTOK=${rest##*:}
        local OUTPATH=$EVAL_DIR/ext_${TASK}_${NAME}
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
    sleep 3
}

# Sources first, then merge variants
ext_eval "src_jackrong-v2"       $EXTQ/src_jackrong-v2-Q6_K.gguf
ext_eval "src_continuum-forged"  $EXTQ/src_continuum-forged-Q6_K.gguf
ext_eval "src_jackrong-python"   $EXTQ/src_jackrong-python-Q6_K.gguf
ext_eval "v2e-fisher-darex"      $OUT/gguf/v2e-fisher-darex-Q6_K.gguf
ext_eval "v2g-3src-fisher-darex" $V2G_GGUF

echo ""
echo "=== ALL DONE $(date -Iseconds) ===" | tee -a $LOG_MASTER

# Summary
echo ""
echo "=== EXTENDED EVAL SUMMARY ===" | tee -a $LOG_MASTER
printf "%-28s %8s %8s %8s %8s\n" "model" "GSM8K" "MMLU-Pro" "AIME" "HE-Plus" | tee -a $LOG_MASTER
printf "%-28s %8s %8s %8s %8s\n" "----" "-----" "--------" "----" "-------" | tee -a $LOG_MASTER
for v in src_jackrong-v2 src_continuum-forged src_jackrong-python v2e-fisher-darex v2g-3src-fisher-darex; do
    GS=$(python3 -c "
import json,glob
fs=sorted(glob.glob('$EVAL_DIR/ext_gsm8k_${v}/${v}/results_*.json'))
if not fs: print('-'); exit()
r=json.load(open(fs[0]))['results']
for k,v in r.items():
    if 'gsm8k' in k:
        for mk,mv in v.items():
            if 'exact_match' in mk and isinstance(mv,(int,float)):
                print(f'{mv*100:.1f}'); exit()
print('-')
" 2>/dev/null)
    MP=$(python3 -c "
import json,glob
fs=sorted(glob.glob('$EVAL_DIR/ext_mmlu_pro_${v}/${v}/results_*.json'))
if not fs: print('-'); exit()
r=json.load(open(fs[0]))['results']
for k,v in r.items():
    if 'mmlu_pro' in k:
        for mk,mv in v.items():
            if 'exact_match' in mk and isinstance(mv,(int,float)):
                print(f'{mv*100:.1f}'); exit()
print('-')
" 2>/dev/null)
    AM=$(python3 -c "
import json,glob
fs=sorted(glob.glob('$EVAL_DIR/ext_aime_${v}/${v}/results_*.json'))
if not fs: print('-'); exit()
r=json.load(open(fs[0]))['results']
for k,v in r.items():
    if 'aime' in k:
        for mk,mv in v.items():
            if 'exact_match' in mk and isinstance(mv,(int,float)):
                print(f'{mv*100:.1f}'); exit()
print('-')
" 2>/dev/null)
    HP=$(python3 -c "
import json,glob
fs=sorted(glob.glob('$EVAL_DIR/ext_humaneval_plus_${v}/${v}/results_*.json'))
if not fs: print('-'); exit()
r=json.load(open(fs[0]))['results']
for k,v in r.items():
    if 'humaneval' in k:
        for mk,mv in v.items():
            if 'pass@1' in mk and isinstance(mv,(int,float)):
                print(f'{mv*100:.1f}'); exit()
print('-')
" 2>/dev/null)
    printf "%-28s %8s %8s %8s %8s\n" "$v" "${GS:-N/A}" "${MP:-N/A}" "${AM:-N/A}" "${HP:-N/A}" | tee -a $LOG_MASTER
done
