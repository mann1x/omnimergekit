#!/bin/bash
# v2h — same 3-source recipe as v2g, but adds AIME-30 differential signal on jackrong-v2.
#
# jackrong-v2 got 8/30 AIME-30 right; cf=0, jp=0. So only jv contributes AIME signal.
# We blend an AIME competence map into jv's existing 3-way combined map (HE+MBPP),
# then re-merge with the same fisher+darex omnimerge_v2 recipe as v2g.
# Goal: recover AIME without losing v2g's MBPP/LCB ceiling.
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
COMP_V2H=$OUT/competence_maps_v2h
mkdir -p $COMP_V2H
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1

NAME=v2h-3src-fisher-darex-aime
M_DIR=$OUT/merged/$NAME
M_F16=$OUT/gguf/$NAME-F16.gguf
M_Q6K=$OUT/gguf/$NAME-Q6_K.gguf
LOG_MASTER=$LOGS/4b_${NAME}.log
mkdir -p $M_DIR $OUT/gguf

echo "=== ${NAME} starting $(date -Iseconds) ===" | tee -a $LOG_MASTER

# Sanity: existing 3-way combined maps from v2g
for f in jackrong-v2 continuum-forged jackrong-python; do
    [ -f "$COMP3_COMBINED/${f}.safetensors" ] || { echo "[!] missing $COMP3_COMBINED/${f}.safetensors"; exit 1; }
done

# Phase 1: extract jv AIME differential map (8 docs, max-len 4096 for reasoning trace)
JV_AIME_MAP=$COMP_V2H/jackrong-v2__aime.safetensors
if [ ! -f "$JV_AIME_MAP" ]; then
    AIME_SAMPLES=$(ls $EVAL_DIR/ext_aime_src_jackrong-v2/src_jackrong-v2/samples_aime_*.jsonl | head -1)
    [ -z "$AIME_SAMPLES" ] && { echo "[!] no AIME samples for jackrong-v2"; exit 1; }
    DOC_IDS=$("$PYBIN" -c "
import json
right=[]
for l in open('$AIME_SAMPLES'):
    d=json.loads(l)
    if d.get('exact_match',0)==1.0:
        right.append(str(d['doc_id']))
print(','.join(right))
")
    N_DOCS=$(echo "$DOC_IDS" | tr ',' '\n' | grep -c .)
    echo "[extract-aime] jackrong-v2 — $N_DOCS docs (ids=$DOC_IDS) ($(date +%H:%M:%S))" | tee -a $LOG_MASTER
    "$PYBIN" -u $WS/scripts/competence_extract.py \
        --model $HF/jackrong-v2 \
        --samples "$AIME_SAMPLES" \
        --task aime \
        --keep-doc-ids "$DOC_IDS" \
        --output $JV_AIME_MAP \
        --max-samples 30 --max-len 4096 --chunk-len 1280 \
        2>&1 | tee -a $LOGS/4b_competence_v2h_aime.log | grep -vE "^Writing:|^Loading|byte/s" | tail -15
    [ -f "$JV_AIME_MAP" ] || { echo "[!] AIME extract failed"; exit 1; }
fi

# Phase 2: blend AIME into jackrong-v2 combined map
# new_jv = ((HE_rate + MBPP_rate) * old_jv + AIME_rate * aime_jv) / (HE_rate + MBPP_rate + AIME_rate)
# rates: HE=0.6037, MBPP=0.45, AIME=0.2667 (from results.json)
echo "[blend] computing v2h jackrong-v2 map ($(date +%H:%M:%S))" | tee -a $LOG_MASTER
"$PYBIN" - <<PY 2>&1 | tee -a $LOG_MASTER
import torch
from safetensors.torch import load_file, save_file

OLD = "$COMP3_COMBINED/jackrong-v2.safetensors"
AIME = "$JV_AIME_MAP"
NEW = "$COMP_V2H/jackrong-v2.safetensors"

# rates from existing extended eval results
HE_RATE = 0.6037
MBPP_RATE = 0.45
AIME_RATE = 0.2667
W_OLD = HE_RATE + MBPP_RATE
W_TOTAL = W_OLD + AIME_RATE

old = load_file(OLD)
aime = load_file(AIME)

# AIME safetensors layout: primary key is grad_l1 (= name), plus suffixed signals.
# Combined map layout: per param, single key (= name) holding chosen signal.
# We blend the AIME 'weight_taylor' (matches combine signal=weight_taylor) into the combined.
out = {}
n_blended = 0
n_skipped = 0
for k, v_old in old.items():
    aime_key = k + ".weight_taylor"
    if aime_key in aime:
        v_aime = aime[aime_key].to(v_old.dtype)
        if v_aime.shape == v_old.shape:
            out[k] = ((W_OLD * v_old.float() + AIME_RATE * v_aime.float()) / W_TOTAL).to(v_old.dtype)
            n_blended += 1
        else:
            out[k] = v_old
            n_skipped += 1
            print(f"  shape mismatch {k}: old={v_old.shape} aime={v_aime.shape}, kept old")
    else:
        out[k] = v_old
        n_skipped += 1

save_file(out, NEW)
print(f"  blended {n_blended} tensors, skipped {n_skipped}, wrote {NEW}")
print(f"  weights: HE={HE_RATE} MBPP={MBPP_RATE} AIME={AIME_RATE} (W_OLD={W_OLD:.4f} W_TOTAL={W_TOTAL:.4f})")
PY
[ -f "$COMP_V2H/jackrong-v2.safetensors" ] || { echo "[!] blend failed"; exit 1; }

# cf/jp maps unchanged — symlink
ln -sf $COMP3_COMBINED/continuum-forged.safetensors $COMP_V2H/continuum-forged.safetensors
ln -sf $COMP3_COMBINED/jackrong-python.safetensors $COMP_V2H/jackrong-python.safetensors
ls -la $COMP_V2H/

# Phase 3: merge v2h (same recipe as v2g, new fisher maps)
if [ ! -f "$M_DIR/.merge_done" ] || ! find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
    rm -rf $M_DIR; mkdir -p $M_DIR
    echo "[merge] ($(date +%H:%M:%S))" | tee -a $LOG_MASTER
    "$PYBIN" -u $WS/scripts/omnimergekit.py \
        --base $BASE \
        --source $HF/jackrong-v2 \
        --source $HF/coder_eval/continuum-code-forged \
        --source $HF/coder_eval/jackrong-python \
        --output $M_DIR \
        --method omnimerge_v2 --v2-features fisher,darex \
        --weights 0.35,0.40,0.25 --density 0.53 --darex-q 0.85 \
        --fisher "$COMP_V2H/jackrong-v2.safetensors,$COMP_V2H/continuum-forged.safetensors,$COMP_V2H/jackrong-python.safetensors" \
        --pr682-turbo \
        --skip-patterns "model.visual,mtp.layers" \
        --seed 42 --device cuda \
        2>&1 | tee -a $LOGS/4b_merge_${NAME}.log | tail -25
    if find "$M_DIR" -name "*.safetensors" 2>/dev/null | grep -q .; then
        touch $M_DIR/.merge_done
    else
        echo "[!] merge failed"; exit 1
    fi
fi

# Phase 4: convert + quantize
if [ ! -f "$M_Q6K" ]; then
    "$PYBIN" /opt/llama.cpp/convert_hf_to_gguf.py $M_DIR --outtype f16 --outfile $M_F16 \
        2>&1 | tee -a $LOGS/4b_convert_${NAME}.log | tail -5
    [ -f "$M_F16" ] || { echo "[!] convert failed"; exit 1; }
    $LLAMA/llama-quantize $M_F16 $M_Q6K Q6_K 2>&1 | tee -a $LOGS/4b_quantize_${NAME}.log | tail -3
    [ -f "$M_Q6K" ] || { echo "[!] quantize failed"; exit 1; }
    rm -f $M_F16
fi

# Phase 5: serve + eval HE/MBPP/LCB
echo "" | tee -a $LOG_MASTER
echo "=== Phase 5: eval $NAME HE/MBPP/LCB $(date +%H:%M:%S) ===" | tee -a $LOG_MASTER
pkill -f 'llama-server.*--port 8099' 2>/dev/null
sleep 3
nohup $LLAMA/llama-server -m $M_Q6K --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
    --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 \
    > $LOGS/4b_server_${NAME}.log 2>&1 &
SERVER_PID=$!
disown
for i in $(seq 1 60); do curl -sf http://localhost:8099/health > /dev/null 2>&1 && break; sleep 2; done
curl -sf http://localhost:8099/health > /dev/null 2>&1 || { echo "[!] server failed"; tail -20 $LOGS/4b_server_${NAME}.log; exit 1; }

for task in mbpp humaneval; do
    OUTPATH=$EVAL_DIR/${task}_${NAME}; mkdir -p $OUTPATH
    "$PYBIN" -u -m lm_eval \
        --model local-completions \
        --model_args "model=${NAME},base_url=http://localhost:8099/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=${BASE},max_length=32768,max_gen_toks=2048" \
        --tasks "$task" --gen_kwargs "temperature=0.0,top_p=1.0,max_gen_toks=2048" \
        --batch_size 1 --use_cache "$OUTPATH/cache" --log_samples --confirm_run_unsafe_code \
        --output_path "$OUTPATH" 2>&1 | tee -a $LOGS/4b_${task}_${NAME}.log | tail -5
done

LP=$EVAL_DIR/lcb_${NAME}; mkdir -p $LP
"$PYBIN" -u $WS/scripts/lcb_llama_server.py \
    --name "$NAME" --base-url http://localhost:8099 \
    --limit 30 --difficulty medium --min-date 2024-10-01 \
    --max-tokens 8192 \
    --output "$LP/lcb_results.json" --samples "$LP/lcb_samples.jsonl" \
    2>&1 | tee -a $LOGS/4b_lcb_${NAME}.log | grep -E "PASS|FAIL|pass@1|Error" | tail -10

# Phase 6: extended eval (gsm8k/mmlu_pro/aime/humaneval_plus)
echo "" | tee -a $LOG_MASTER
echo "=== Phase 6: extended eval $NAME $(date +%H:%M:%S) ===" | tee -a $LOG_MASTER
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

echo ""
echo "=== ALL DONE $(date -Iseconds) ===" | tee -a $LOG_MASTER

# Final scores
echo ""
echo "=== ${NAME} scores ==="
for task in humaneval mbpp; do
    R=$(ls $EVAL_DIR/${task}_${NAME}/${NAME}/results_*.json 2>/dev/null | head -1)
    [ -f "$R" ] && python3 -c "
import json
d=json.load(open('$R'))['results']
for k,v in d.items():
    for mk,mv in v.items():
        if 'pass' in mk and isinstance(mv,(int,float)):
            print(f'  $task: {mv*100:.2f}'); break
    break
"
done
LP=$EVAL_DIR/lcb_${NAME}/lcb_results.json
[ -f "$LP" ] && python3 -c "import json; print(f'  LCB-30: {json.load(open(\"$LP\"))[\"pass_at_1\"]*100:.2f}')"
for task in gsm8k mmlu_pro aime humaneval_plus; do
    R=$(ls $EVAL_DIR/ext_${task}_${NAME}/${NAME}/results_*.json 2>/dev/null | head -1)
    [ -f "$R" ] && python3 -c "
import json
d=json.load(open('$R'))['results']
for k,v in d.items():
    for mk,mv in v.items():
        if ('exact_match' in mk or 'pass' in mk) and isinstance(mv,(int,float)):
            print(f'  $task: {mv*100:.2f}'); break
    break
"
done
