#!/usr/bin/env bash
# Publication eval suite for Gemma 4 26B-A4B family (98e v3, 98e v4, 128e).
# Implements EVAL_PROTOCOL §v2.7 canonical per-bench backend split:
#
#   Bench          Backend     Quant       Why
#   -------------- ----------- ----------- ----------------------------------
#   LCB-55         vLLM        NVFP4A16    Gemma 4 MoE+llama.cpp distrust
#   MBPP-500       vLLM        NVFP4A16    inconsistent llama.cpp results
#   GPQA-Diamond   vLLM        NVFP4A16    reasoning on scorer-validated path
#   HE+            llama.cpp   Q6_K        trusted, 3× faster
#   GSM8K-100      llama.cpp   Q6_K        trusted, 3× faster
#   MMLU-Pro-200   llama.cpp   Q6_K        trusted, 3× faster
#   AIME-30        llama.cpp   Q6_K        trusted, 3× faster
#
# HumanEval-164 is intentionally EXCLUDED — already published from
# llama.cpp Q6_K runs, do not re-run.
#
# This script does NOT chain across variants in one process — it runs
# all benches for one variant, exits, and the caller (or another
# invocation) handles the next variant. That keeps the GPU lifecycle
# clean (vLLM teardown + llama-server startup per variant).
#
# Usage:
#   bash run_a4b_publication_suite.sh <variant> [phase]
#     variant := v3 | v4 | 128e
#     phase   := all | vllm | llama  (default: all)
#
# Examples:
#   bash run_a4b_publication_suite.sh v3            # full suite for 98e v3
#   bash run_a4b_publication_suite.sh v4 vllm       # only LCB+MBPP+GPQA for v4
#   bash run_a4b_publication_suite.sh 128e llama    # only HE+/GSM8K/MMLU-Pro/AIME
set -uo pipefail

VARIANT="${1:?usage: $0 <v3|v4|128e> [all|vllm|llama]}"
PHASE="${2:-all}"

WS="${WS:-/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models}"
OMK="${OMK:-/shared/dev/omnimergekit}"
RESULTS="${RESULTS:-$WS/eval_results_publication}"
LOGS="${LOGS:-$WS/logs}"
PORT="${PORT:-8195}"
LLAMA="${LLAMA:-/opt/llama.cpp}"

mkdir -p "$RESULTS" "$LOGS"

# ── variant paths ────────────────────────────────────────────────────────
case "$VARIANT" in
    v3)
        BF16="$WS/google/gemma-4-A4B-98e-v3-it"
        NVFP4="$WS/google/gemma-4-A4B-98e-v3-it-NVFP4A16"
        GGUF="$WS/google/gemma-4-A4B-98e-v3-Q6_K.gguf"
        NAME="98e_v3"
        ;;
    v4)
        BF16="$WS/google/gemma-4-A4B-98e-v4-it"
        NVFP4="$WS/google/gemma-4-A4B-98e-v4-it-NVFP4A16"
        GGUF="$WS/google/gemma-4-A4B-98e-v4-Q6_K.gguf"
        NAME="98e_v4"
        ;;
    128e)
        BF16="$WS/google/gemma-4-26B-A4B-it"
        NVFP4="$WS/google/Gemma-4-26B-A4B-it-NVFP4A16"
        GGUF="$WS/google/gemma-4-26B-A4B-it-Q6_K.gguf"
        NAME="128e"
        ;;
    *)  echo "ERR: unknown variant '$VARIANT'"; exit 2 ;;
esac

TOK="$WS/google/gemma-4-26B-A4B-it"   # tokenizer must be the original 128e dir

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOGS/pub_${NAME}.log"; }
log "=== publication suite: $NAME (phase=$PHASE) ==="
log "BF16 src: $BF16"
log "NVFP4:    $NVFP4"
log "GGUF:     $GGUF"

# ── vLLM phase: ensure NVFP4A16 exists, then LCB-55, MBPP-500, GPQA ───────
do_vllm() {
    if [[ ! -f "$NVFP4/config.json" ]]; then
        [[ -f "$BF16/config.json" ]] || { log "ABORT: no BF16 src at $BF16"; return 1; }
        log "[$NAME] quantizing $BF16 -> $NVFP4 (NVFP4A16 via modelopt)"
        /root/anaconda3/envs/modelopt/bin/python "$OMK/scripts/quantize_any.py" \
            --src "$BF16" --dst "$NVFP4" --method nvfp4a16 \
            2>&1 | tee -a "$LOGS/nvfp4_quant_${NAME}.log"
        [[ -f "$NVFP4/config.json" ]] || { log "[$NAME] QUANT FAILED"; return 1; }
    fi
    local sname="${NAME}_nvfp4a16"
    for tpl in lcb_medium_55 mbpp_full gpqa_diamond_full; do
        log "[$NAME] vllm bench: $tpl"
        python3 "$OMK/eval/omk_eval.py" \
            --model "$NVFP4" \
            --template "$tpl" \
            --backend vllm \
            --quant auto \
            --port "$PORT" \
            --results-dir "$RESULTS" \
            --served-name "$sname" \
            --tokenizer "$TOK" \
            2>&1 | tee -a "$LOGS/pub_${NAME}_${tpl}.log"
        # Force-kill any lingering vllm server before next bench (omk_eval
        # tears down, but we belt-and-suspender it on the publication run).
        pkill -f "vllm.*--port $PORT" 2>/dev/null || true
        sleep 5
    done
}

# ── llama.cpp phase: HE+, GSM8K-100, MMLU-Pro-200, AIME-30 ────────────────
ensure_q6k() {
    [[ -f "$GGUF" ]] && return 0
    [[ -f "$BF16/config.json" ]] || { log "ABORT: no BF16 src at $BF16"; return 1; }
    local f16="${GGUF%.Q6_K.gguf}.F16.gguf"
    log "[$NAME] convert BF16 -> F16: $BF16 -> $f16"
    python3 "$LLAMA/convert_hf_to_gguf.py" "$BF16" --outfile "$f16" --outtype f16 \
        2>&1 | tee -a "$LOGS/q6k_${NAME}.log" | tail -3
    [[ -f "$f16" ]] || { log "[$NAME] F16 FAILED"; return 1; }
    log "[$NAME] quantize F16 -> Q6_K"
    "$LLAMA/build/bin/llama-quantize" "$f16" "$GGUF" Q6_K \
        2>&1 | tee -a "$LOGS/q6k_${NAME}.log" | tail -3
    rm -f "$f16"
    [[ -f "$GGUF" ]]
}

do_llama() {
    ensure_q6k || return 1
    local sname="${NAME}_q6k"
    # Port 8099 to avoid colliding with any vllm leftover on 8195.
    for tpl in humanevalplus_full gsm8k_100 mmlu_pro_200 aime_30; do
        log "[$NAME] llama bench: $tpl"
        python3 "$OMK/eval/omk_eval.py" \
            --model "$GGUF" \
            --template "$tpl" \
            --backend llama \
            --port 8099 \
            --results-dir "$RESULTS" \
            --served-name "$sname" \
            --tokenizer "$TOK" \
            2>&1 | tee -a "$LOGS/pub_${NAME}_${tpl}.log"
        pkill -f "llama-server.*--port 8099" 2>/dev/null || true
        sleep 3
    done
}

case "$PHASE" in
    all)   do_vllm; do_llama ;;
    vllm)  do_vllm ;;
    llama) do_llama ;;
    *)     log "ERR: unknown phase '$PHASE'"; exit 2 ;;
esac

log "=== publication suite for $NAME ($PHASE) DONE ==="
log "Results: $RESULTS/<bench>/$NAME_*/summary.json"
log "Roll up to a card table with:"
log "  python3 $OMK/eval/omk_summarize.py $RESULTS/<bench>/*"
