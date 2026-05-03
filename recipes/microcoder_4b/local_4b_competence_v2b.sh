#!/bin/bash
# Competence-driven merge v2b — DIFFERENTIAL signal.
#
# Key change vs v2a: each source's competence map is computed ONLY on problems
# THIS source uniquely solved (vs the OTHER source). High contrast → fisher
# election actually flips per-element.
#
# 2-source: jackrong-v2 + continuum-forged (v1b recipe).
# Differential sets (computed inline below):
#   jackrong-v2:      HE 15 docs + MBPP 27 docs   = 42 samples
#   continuum-forged: HE 13 docs + MBPP 69 docs   = 82 samples
set -uo pipefail

WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
HF=$WS/hf_models_4b
OUT=$WS/4b_phase1
LOGS=$WS/logs
LLAMA=/opt/llama.cpp/build/bin
PYBIN=/shared/dev/lightseek/.venv/bin/python
BASE=$HF/Qwen3.5-4B

NAME=${NAME:-v2b-competence-diff}
COMP_DIR=$OUT/competence_maps_diff
COMP_COMBINED=$COMP_DIR/combined
M_DIR=$OUT/merged/$NAME
M_F16=$OUT/gguf/$NAME-F16.gguf
M_Q6K=$OUT/gguf/$NAME-Q6_K.gguf
LOG_MASTER=$LOGS/4b_${NAME}.log
EVAL_DIR=$OUT/eval_results
mkdir -p $COMP_DIR $COMP_COMBINED $M_DIR $OUT/gguf $LOGS $EVAL_DIR
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1

echo "=== ${NAME} starting $(date -Iseconds) ===" | tee -a $LOG_MASTER

# Phase 0: compute differential doc-id sets for the 2-source pair
DIFFSETS=$COMP_DIR/_diffsets.json
echo "[diffsets] computing pairwise disagreement (jackrong-v2 vs continuum-forged)" | tee -a $LOG_MASTER
"$PYBIN" -u <<PY 2>&1 | tee -a $LOG_MASTER
import json, glob, sys
EVAL = "$WS/4b_phase1/eval_results"
SRC_BN = {"jackrong-v2": "jackrong-v2", "continuum-forged": "continuum-code-forged"}

def load_pass(task_name, bn, pass_key):
    fs = sorted(glob.glob(f"{EVAL}/{task_name}_{bn}/{bn}/samples_*.jsonl"))
    if not fs: return {}
    return {str(json.loads(l)["doc_id"]): float(json.loads(l).get(pass_key, 0)) >= 1.0
            for l in open(fs[0])}

result = {}
for task, key in [("humaneval", "pass@1"), ("mbpp", "pass_at_1")]:
    pa = load_pass(task, SRC_BN["jackrong-v2"], key)
    pb = load_pass(task, SRC_BN["continuum-forged"], key)
    common = set(pa) & set(pb)
    a_only = sorted(d for d in common if pa[d] and not pb[d])
    b_only = sorted(d for d in common if pb[d] and not pa[d])
    short = "he" if task == "humaneval" else "mbpp"
    result[f"jackrong-v2__{short}"] = a_only
    result[f"continuum-forged__{short}"] = b_only
    print(f"  {task}: jackrong-v2_only={len(a_only)} continuum-forged_only={len(b_only)}")

with open("$DIFFSETS", "w") as f:
    json.dump(result, f, indent=2)
print(f"  wrote $DIFFSETS")
PY

# Phase 1: extract differential competence maps (4 files: 2 sources × 2 tasks)
declare -A SRC_HF=(
    [jackrong-v2]=$HF/jackrong-v2
    [continuum-forged]=$HF/coder_eval/continuum-code-forged
)
declare -A EVAL_BN=(
    [jackrong-v2]=jackrong-v2
    [continuum-forged]=continuum-code-forged
)

for src in jackrong-v2 continuum-forged; do
    bn=${EVAL_BN[$src]}
    for task in he mbpp; do
        OUT_F=$COMP_DIR/${src}__${task}.safetensors
        if [ -f "$OUT_F" ]; then
            echo "[extract] $src $task — already done, skip" | tee -a $LOG_MASTER
            continue
        fi
        case $task in
            he)   GLOB="$OUT/eval_results/humaneval_${bn}/${bn}/samples_humaneval_*.jsonl" ;;
            mbpp) GLOB="$OUT/eval_results/mbpp_${bn}/${bn}/samples_mbpp_*.jsonl" ;;
        esac
        # Pull doc-id list from JSON
        DOC_IDS=$("$PYBIN" -c "import json; d=json.load(open('$DIFFSETS')); print(','.join(d['${src}__${task}']))")
        N_DOCS=$(echo "$DOC_IDS" | tr ',' '\n' | grep -c .)
        echo "[extract] $src $task — $N_DOCS unique docs ($(date +%H:%M:%S))" | tee -a $LOG_MASTER
        "$PYBIN" -u $WS/scripts/competence_extract.py \
            --model ${SRC_HF[$src]} \
            --samples "$GLOB" \
            --task $task \
            --keep-doc-ids "$DOC_IDS" \
            --output $OUT_F \
            --max-samples 200 --max-len 512 \
            2>&1 | tee -a $LOGS/4b_competence_diff_${src}_${task}.log | grep -vE "^Writing:|^Loading|byte/s" | tail -30
        if [ ! -f "$OUT_F" ]; then
            echo "[!] extract failed: $src $task" | tee -a $LOG_MASTER
            exit 1
        fi
    done
done

# Phase 2: combine. Use raw rates per task (we don't subtract base; the diffset
# IS the differential). signal=weight_taylor.
echo "[combine] ($(date +%H:%M:%S))" | tee -a $LOG_MASTER
MAP_ARGS=()
for src in jackrong-v2 continuum-forged; do
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
    2>&1 | tee -a $LOGS/4b_competence_diff_combine.log
ls -la $COMP_COMBINED | tee -a $LOG_MASTER
# Reclaim disk
rm -f $COMP_DIR/*__he.safetensors $COMP_DIR/*__mbpp.safetensors

# Phase 3: merge with v1b recipe + new differential fisher maps
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

# Phase 4: convert + quantize
if [ ! -f "$M_Q6K" ]; then
    python3 /opt/llama.cpp/convert_hf_to_gguf.py $M_DIR --outtype f16 --outfile $M_F16 \
        2>&1 | tee -a $LOGS/4b_convert_${NAME}.log | tail -8
    [ -f "$M_F16" ] || { echo "[!] $NAME convert failed"; exit 1; }
    $LLAMA/llama-quantize $M_F16 $M_Q6K Q6_K 2>&1 | tee -a $LOGS/4b_quantize_${NAME}.log | tail -3
    [ -f "$M_Q6K" ] || { echo "[!] $NAME quantize failed"; exit 1; }
    rm -f $M_F16
fi

# Phase 5: serve + eval
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
