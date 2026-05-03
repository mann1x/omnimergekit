#!/bin/bash
# Phase 2 of the Qwen3.5-27B Omnimerge pipeline:
#   1) Run mergekit DARE-TIES 4-way using qwen3_5_omnimerge_dare_ties.yml
#      (HF sources already downloaded in /workspace/hf_models/ by Phase 1)
#   2) Convert merged HF → F16 GGUF → Q6_K GGUF
#   3) HumanEval pass@1 on the merged model
#   4) Upload merged HF model to ManniX-ITA/Qwen3.5-27B-Omnimerge
#   5) Upload Q6_K GGUF to ManniX-ITA/Qwen3.5-27B-Omnimerge-GGUF
#
# Invoke on pod (as root) after Phase 1 is complete:
#   HF_TOKEN=hf_xxx bash pod_qwen3_5_merge_and_eval.sh
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN required}"

WORKDIR=/workspace
HF_ROOT=$WORKDIR/hf_models
MERGED=$WORKDIR/merged_omnimerge
GGUF_ROOT=$WORKDIR/gguf
LLAMA_BIN=/opt/llama.cpp/build/bin
CFG=$WORKDIR/qwen3_5_omnimerge_dare_ties.yml
LOG=$WORKDIR/merge_and_eval.log

mkdir -p "$GGUF_ROOT"

echo "===== $(date) PHASE 2: MERGE + EVAL + PUBLISH =====" | tee -a "$LOG"

# --- Step 1+2: verify sources + run merge (skipped entirely if merged dir already exists) ---
# mergekit 0.1.4 + git-main do NOT support Qwen3.5 (Qwen3_5ForConditionalGeneration,
# model_type='qwen3_5' with hybrid linear_attn/SSM layers). The fallback is the custom
# merger scripts/dare_ties_merge.py which must be run externally BEFORE this script.
# If $MERGED/config.json exists we assume the merge is already done and jump straight
# to convert → quantize → eval.
if [[ ! -f "$MERGED/config.json" ]]; then
    for m in Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled \
             DavidAU/Qwen3.5-27B-Gemini3-Pro-High-Reasoning-Compact-Thinking \
             ConicCat/Qwen3.5-27B-Writer-V2 \
             ValiantLabs/Qwen3.5-27B-Esper3.1; do
        name=$(basename "$m")
        if [[ ! -f "$HF_ROOT/$name/config.json" ]]; then
            echo "ERROR: missing $HF_ROOT/$name — run Phase 1 first (or use dare_ties_merge.py directly)" | tee -a "$LOG"
            exit 1
        fi
    done

    echo | tee -a "$LOG"
    echo "=== mergekit DARE-TIES 4-way ===" | tee -a "$LOG"
    # Use --lazy-unpickle to reduce peak RAM; --cuda only speeds attention but we're merging
    # on CPU anyway for large 27B models
    HF_HOME=$HF_ROOT mergekit-yaml "$CFG" "$MERGED" \
        --allow-crimes \
        --lazy-unpickle \
        --out-shard-size 5B \
        --clone-tensors \
        --copy-tokenizer 2>&1 | tee -a "$LOG"
else
    echo | tee -a "$LOG"
    echo "=== merged dir already exists, skipping merge stage ===" | tee -a "$LOG"
    echo "  (if this is the custom dare_ties_merge.py output, that's the intended flow)" | tee -a "$LOG"
fi

du -sh "$MERGED" | tee -a "$LOG"

# --- Step 3: convert merged HF → F16 GGUF ---
F16=$GGUF_ROOT/Qwen3.5-27B-Omnimerge-F16.gguf
Q6K=$GGUF_ROOT/Qwen3.5-27B-Omnimerge-Q6_K.gguf

if [[ ! -f "$F16" && ! -f "$Q6K" ]]; then
    echo | tee -a "$LOG"
    echo "=== convert merged → F16 GGUF ===" | tee -a "$LOG"
    python3 /opt/llama.cpp/convert_hf_to_gguf.py "$MERGED" \
        --outfile "$F16" --outtype f16 2>&1 | tail -10 | tee -a "$LOG"
fi

if [[ ! -f "$Q6K" ]]; then
    echo | tee -a "$LOG"
    echo "=== quantize F16 → Q6_K ===" | tee -a "$LOG"
    "$LLAMA_BIN/llama-quantize" "$F16" "$Q6K" Q6_K 2>&1 | tail -10 | tee -a "$LOG"
fi

# --- Step 4: HumanEval pass@1 on merged Q6_K ---
# Uses raw text-completion API (local-completions) to avoid Qwen3.5 chat-mode
# markdown-fence trap that made every prior chat-completion run score 0%.
# See bug-015 for the full story. This matches pod_qwen3_5_eval_sources.sh
# exactly so the merge number is directly comparable to the source numbers.
# MANDATORY: --use_cache + --log_samples + post-run sanity check.
echo | tee -a "$LOG"
echo "=== HumanEval merged Q6_K (raw completions) ===" | tee -a "$LOG"
"$LLAMA_BIN/llama-server" \
    -m "$Q6K" \
    --port 8099 -c 32768 -ngl 99 --no-warmup --parallel 1 \
    --seed 42 \
    > $WORKDIR/merged_server.log 2>&1 &
SPID=$!
for i in $(seq 1 120); do
    curl -fsS http://localhost:8099/health 2>/dev/null | grep -q ok && break
    kill -0 $SPID 2>/dev/null || { echo "  server died during startup" | tee -a "$LOG"; break; }
    sleep 1
done

export HF_ALLOW_CODE_EVAL=1
mkdir -p "$WORKDIR/humaneval_cache"
CACHE_PREFIX="$WORKDIR/humaneval_cache/Omnimerge"
rm -rf "$WORKDIR/humaneval_merged" 2>/dev/null || true

MAX_RETRIES=${MAX_RETRIES:-6}
RETRY_DELAY=${RETRY_DELAY:-15}
EVAL_OK=0
for attempt in $(seq 1 $MAX_RETRIES); do
    echo "  [eval attempt $attempt/$MAX_RETRIES] lm_eval humaneval..." | tee -a "$LOG"
    set +e
    lm_eval \
        --model local-completions \
        --model_args "model=Omnimerge,base_url=http://localhost:8099/v1/completions,num_concurrent=1,tokenizer_backend=huggingface,tokenizer=$MERGED,max_gen_toks=1024" \
        --tasks humaneval \
        --batch_size 1 \
        --confirm_run_unsafe_code \
        --use_cache "$CACHE_PREFIX" \
        --log_samples \
        --gen_kwargs "temperature=0.0,max_gen_toks=1024" \
        --output_path "$WORKDIR/humaneval_merged" 2>&1 | tee -a "$LOG"
    RC=${PIPESTATUS[0]}
    set -e
    if [[ $RC -eq 0 ]]; then
        EVAL_OK=1
        break
    fi
    echo "  lm_eval exited $RC on attempt $attempt, retrying in ${RETRY_DELAY}s..." | tee -a "$LOG"
    kill $SPID 2>/dev/null || true
    wait $SPID 2>/dev/null || true
    sleep $RETRY_DELAY
    "$LLAMA_BIN/llama-server" \
        -m "$Q6K" \
        --port 8099 -c 32768 -ngl 99 --no-warmup --parallel 1 \
        --seed 42 \
        >> $WORKDIR/merged_server.log 2>&1 &
    SPID=$!
    for i in $(seq 1 120); do
        curl -fsS http://localhost:8099/health 2>/dev/null | grep -q ok && break
        kill -0 $SPID 2>/dev/null || { echo "  retry server died" | tee -a "$LOG"; break; }
        sleep 1
    done
done

if [[ $EVAL_OK -ne 1 ]]; then
    echo "  HumanEval FAILED for Omnimerge after $MAX_RETRIES attempts" | tee -a "$LOG"
fi

# MANDATORY sanity check: validate samples jsonl for real Python code
SAMP=$(ls "$WORKDIR/humaneval_merged"/**/samples_humaneval_*.jsonl 2>/dev/null | head -1 || true)
RES=$(ls "$WORKDIR/humaneval_merged"/**/results_*.json 2>/dev/null | head -1 || true)
if [[ -n "$SAMP" && -n "$RES" ]]; then
    python3 - <<PY | tee -a "$LOG"
import json
samp, res = "$SAMP", "$RES"
with open(res) as f:
    d = json.load(f)
r = list(d["results"].values())[0]
score = next((v for k,v in r.items() if "pass" in k.lower() and "stderr" not in k), None)
total = sum(1 for _ in open(samp))
bad = fence = empty = 0
with open(samp) as f:
    for line in f:
        s = json.loads(line)
        gen = s.get("resps", [[""]])[0][0] if s.get("resps") else ""
        if not gen.strip(): empty += 1
        if "\`\`\`" in gen: fence += 1
        if len(gen.strip()) < 5: bad += 1
print(f"  [sanity] Omnimerge: pass@1={score}  samples={total}  empty={empty}  markdown-fence={fence}  <5chars={bad}")
PY
else
    echo "  [sanity] Omnimerge: NO samples/results file found" | tee -a "$LOG"
fi

kill $SPID 2>/dev/null || true
wait $SPID 2>/dev/null || true

# --- Step 5: print comparison table ---
echo | tee -a "$LOG"
echo "=== COMPARISON ===" | tee -a "$LOG"
for d in $WORKDIR/humaneval_*/; do
    name=$(basename "$d" | sed 's/humaneval_//')
    result_file=$(ls $d**/results_*.json 2>/dev/null | head -1 || true)
    if [[ -n "$result_file" ]]; then
        score=$(python3 -c "import json; d = json.load(open('$result_file')); r = list(d['results'].values())[0]; print(next((f'{v*100:.2f}%' for k,v in r.items() if 'pass' in k.lower() and 'stderr' not in k), 'N/A'))")
        echo "  $name: pass@1 = $score" | tee -a "$LOG"
    fi
done

echo | tee -a "$LOG"
echo "===== $(date) MERGE+EVAL DONE — review before upload =====" | tee -a "$LOG"
echo "To upload after review:"
echo "  bash pod_qwen3_5_publish.sh"
