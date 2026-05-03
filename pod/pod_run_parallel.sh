#!/bin/bash
# Parallel pod run: starts 109e GPQA on GPU 0 IMMEDIATELY (no surgery
# needed — direct GGUF download) while 120e v3 surgery+quant runs in
# parallel. As soon as 120e v3 Q6_K is ready, GPU 1 GPQA starts.
#
# Assumes llama.cpp is already built at /opt/llama.cpp/build/bin/.
set -uo pipefail

WORK="/workspace"
mkdir -p "$WORK" && cd "$WORK"

export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set}"
export HUGGINGFACE_TOKEN="$HF_TOKEN"
export HF_HOME="$WORK/.hf_cache"
mkdir -p "$HF_HOME"

LLAMA="/opt/llama.cpp/build/bin/llama-server"
QUANTIZE="/opt/llama.cpp/build/bin/llama-quantize"

if [[ ! -x "$LLAMA" ]] || [[ ! -x "$QUANTIZE" ]]; then
    echo "ERROR: llama-server or llama-quantize not built"
    exit 1
fi

#==============================================================
# 0. Install Python deps (if not already)
#==============================================================
echo "===== installing Python deps ====="
pip install --quiet --upgrade pip
pip install --quiet huggingface_hub transformers safetensors tqdm requests datasets
pip install --quiet "lm-eval[api] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git"
hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>&1 | tail -3 || true

#==============================================================
# 1. Re-embed scripts (in case workspace was wiped)
#==============================================================
mkdir -p scripts
if [[ ! -f scripts/hybrid_120e_drop_map.json ]] || [[ ! -f scripts/expert_drop.py ]]; then
    echo "ERROR: scripts/hybrid_120e_drop_map.json or scripts/expert_drop.py missing"
    echo "       upload pod_setup.sh first to populate them"
    exit 1
fi

LM_EVAL_COMMON='--tasks gpqa_diamond_cot_zeroshot --apply_chat_template --batch_size 1 --gen_kwargs temperature=1.0,top_p=0.95,max_gen_toks=24576 --log_samples'
SERVER_COMMON="-c 32768 -t 16 -ngl 99 --no-warmup --reasoning-format deepseek --reasoning-budget 16384 --temp 1.0 --top-p 0.95 --top-k 64 --seed 42"

#==============================================================
# 2. PARALLEL: 109e path on GPU 0
#==============================================================
(
    set -euo pipefail
    LABEL="109e"
    LOG="$WORK/path_109e.log"
    exec > "$LOG" 2>&1
    echo "===== 109e path start: $(date) ====="

    if [[ ! -f gemma-4-A4B-109e-Q6_K.gguf ]]; then
        echo "  -> downloading 109e Q6_K direct from HF..."
        hf download ManniX-ITA/gemma-4-A4B-109e-it-GGUF \
            gemma-4-A4B-109e-it-Q6_K.gguf \
            --local-dir . \
            --max-workers 4 2>&1 | tail -5
        mv -f gemma-4-A4B-109e-it-Q6_K.gguf gemma-4-A4B-109e-Q6_K.gguf 2>/dev/null || true
    fi
    ls -lh gemma-4-A4B-109e-Q6_K.gguf

    echo "  -> starting llama-server on GPU 0 (port 8099)..."
    CUDA_VISIBLE_DEVICES=0 $LLAMA -m gemma-4-A4B-109e-Q6_K.gguf --port 8099 $SERVER_COMMON \
        >llama_109e.log 2>&1 &
    SPID=$!
    disown $SPID 2>/dev/null || true

    for i in $(seq 1 240); do
        if curl -fsS http://localhost:8099/health 2>/dev/null | grep -q ok; then
            echo "  server ready (pid $SPID)"
            break
        fi
        if ! kill -0 $SPID 2>/dev/null; then
            echo "  ERROR: 109e server died"
            tail -20 llama_109e.log
            exit 1
        fi
        sleep 1
    done

    echo "  -> launching lm_eval on 109e..."
    mkdir -p eval_results
    lm_eval --model local-chat-completions \
        --model_args "model=109e,base_url=http://localhost:8099/v1/chat/completions,num_concurrent=1,tokenizer_backend=huggingface,tokenizer=google/gemma-4-26B-A4B-it,max_gen_toks=24576" \
        $LM_EVAL_COMMON \
        --output_path eval_results/109e

    echo "===== 109e path done: $(date) ====="
    kill $SPID 2>/dev/null || true
) &
PID109=$!
echo "  109e path launched, pid $PID109 (log: path_109e.log)"

#==============================================================
# 3. PARALLEL: 120e v3 path on GPU 1
#==============================================================
(
    set -euo pipefail
    LABEL="120e_v3"
    LOG="$WORK/path_120e_v3.log"
    exec > "$LOG" 2>&1
    echo "===== 120e v3 path start: $(date) ====="

    if [[ ! -f gemma-4-A4B-120e-hybrid-Q6_K.gguf ]]; then
        echo "  -> downloading 128e HF (~50 GB)..."
        hf download google/gemma-4-26B-A4B-it \
            --local-dir gemma-4-26B-A4B-it \
            --max-workers 8 2>&1 | tail -5

        echo "  -> running expert_drop for 120e v3..."
        python3 scripts/expert_drop.py \
            --source-dir gemma-4-26B-A4B-it \
            --drop-map scripts/hybrid_120e_drop_map.json \
            --suffix=-hybrid

        rm -rf gemma-4-26B-A4B-it
        df -h "$WORK" | tail -1

        echo "  -> converting 120e v3 to F16..."
        python3 /opt/llama.cpp/convert_hf_to_gguf.py gemma-4-A4B-120e-hybrid \
            --outfile gemma-4-A4B-120e-hybrid-F16.gguf --outtype f16 2>&1 | tail -3

        echo "  -> quantizing 120e v3 to Q6_K..."
        $QUANTIZE gemma-4-A4B-120e-hybrid-F16.gguf gemma-4-A4B-120e-hybrid-Q6_K.gguf Q6_K 2>&1 | tail -3

        rm gemma-4-A4B-120e-hybrid-F16.gguf
        rm -rf gemma-4-A4B-120e-hybrid
        df -h "$WORK" | tail -1
    fi
    ls -lh gemma-4-A4B-120e-hybrid-Q6_K.gguf

    echo "  -> starting llama-server on GPU 1 (port 8100)..."
    CUDA_VISIBLE_DEVICES=1 $LLAMA -m gemma-4-A4B-120e-hybrid-Q6_K.gguf --port 8100 $SERVER_COMMON \
        >llama_120e_v3.log 2>&1 &
    SPID=$!
    disown $SPID 2>/dev/null || true

    for i in $(seq 1 240); do
        if curl -fsS http://localhost:8100/health 2>/dev/null | grep -q ok; then
            echo "  server ready (pid $SPID)"
            break
        fi
        if ! kill -0 $SPID 2>/dev/null; then
            echo "  ERROR: 120e v3 server died"
            tail -20 llama_120e_v3.log
            exit 1
        fi
        sleep 1
    done

    echo "  -> launching lm_eval on 120e v3..."
    mkdir -p eval_results
    lm_eval --model local-chat-completions \
        --model_args "model=120e_v3,base_url=http://localhost:8100/v1/chat/completions,num_concurrent=1,tokenizer_backend=huggingface,tokenizer=google/gemma-4-26B-A4B-it,max_gen_toks=24576" \
        $LM_EVAL_COMMON \
        --output_path eval_results/120e_v3

    echo "===== 120e v3 path done: $(date) ====="
    kill $SPID 2>/dev/null || true
) &
PID120=$!
echo "  120e v3 path launched, pid $PID120 (log: path_120e_v3.log)"

echo
echo "===== both parallel paths running ====="
echo "  109e path:    pid $PID109, log $WORK/path_109e.log"
echo "  120e v3 path: pid $PID120, log $WORK/path_120e_v3.log"
echo
echo "Waiting for both to finish..."

wait $PID109
echo "===== 109e path finished: $(date) ====="
wait $PID120
echo "===== 120e v3 path finished: $(date) ====="

echo
echo "===== both done: $(date) ====="
for d in eval_results/109e eval_results/120e_v3; do
    echo "--- $d ---"
    find "$d" -name "results_*.json" 2>/dev/null | head -1 | while read f; do
        python3 -c "
import json
d = json.load(open('$f'))
for k, v in d.get('results', {}).items():
    print(f'  {k}:')
    for mk, mv in v.items():
        if isinstance(mv, (int, float, str)):
            print(f'    {mk}: {mv}')
"
    done
done
