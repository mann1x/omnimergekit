#!/usr/bin/env bash
# Full MicroCoder benchmark suite via vLLM + omk_eval (protocol v2).
#
# Matches the table on https://huggingface.co/ManniX-ITA/Qwen3.5-4B-MicroCoder:
#   HumanEval, HumanEval+, MBPP, MMLU-Pro (200q stride), GSM8K (100q stride),
#   GPQA Diamond (full 198q), AIME (30q canonical set).
#
# This recipe is intentionally a thin loop over omk_eval.py — all server
# lifecycle, token-stats, sanity check, and SQLite caching live in
# omk_eval.py. Each invocation produces one results dir with samples.jsonl
# + summary.json (per protocol v2 §2.4).
#
# Usage:
#   bash run_full_suite_vllm.sh <model-path-or-hf-id> [served-name] [tag-list]
#     tag-list defaults to "humaneval mbpp humanevalplus gsm8k mmlu_pro gpqa aime"
#     (omit lcb on small coder models — runs separately via the gemma4 recipes)
#
# Example:
#   bash run_full_suite_vllm.sh \
#       /srv/.../backup_models/4b_phase1/microcoder_v2i \
#       microcoder_v2i \
#       humaneval mbpp gsm8k
set -uo pipefail

MODEL="${1:?usage: $0 <model> [served-name] [tags...]}"
NAME="${2:-$(basename "$MODEL")}"
shift 2 2>/dev/null || shift 1
TAGS=("${@:-humaneval mbpp humanevalplus gsm8k mmlu_pro gpqa aime}")
[[ "${TAGS[*]}" == "humaneval mbpp humanevalplus gsm8k mmlu_pro gpqa aime" ]] && \
    TAGS=(humaneval mbpp humanevalplus gsm8k mmlu_pro gpqa aime)

WS="${WS:-/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models}"
OMK="${OMK:-/shared/dev/omnimergekit}"
RESULTS="${RESULTS:-$WS/eval_results_vllm/$NAME}"
LOGS="${LOGS:-$WS/logs}"
PORT="${PORT:-8195}"
QUANT="${QUANT:-auto}"   # auto = let omk_eval detect bf16/fp16/nvfp4/awq/gptq

declare -A TEMPLATE
TEMPLATE[humaneval]=humaneval_full
TEMPLATE[mbpp]=mbpp_full
TEMPLATE[humanevalplus]=humanevalplus_full
TEMPLATE[gsm8k]=gsm8k_100
TEMPLATE[mmlu_pro]=mmlu_pro_200
TEMPLATE[gpqa]=gpqa_diamond_full
TEMPLATE[aime]=aime_30
TEMPLATE[lcb55]=lcb_medium_55
TEMPLATE[lcb30]=lcb_medium_30

mkdir -p "$RESULTS" "$LOGS"
SUITE_LOG="$LOGS/suite_${NAME}_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$SUITE_LOG"; }

log "=== MicroCoder full suite ==="
log "model:    $MODEL"
log "name:     $NAME"
log "tags:     ${TAGS[*]}"
log "results:  $RESULTS"
log "quant:    $QUANT"

# Single vLLM server for the whole suite — omk_eval supports --no-server so
# we don't churn 7×30s startup penalty. The first call brings it up; the
# rest reuse it on the same port.
SERVED=0
for tag in "${TAGS[@]}"; do
    tpl="${TEMPLATE[$tag]:-}"
    if [[ -z "$tpl" ]]; then
        log "WARN: unknown tag '$tag' — skipping"
        continue
    fi
    log "--- $tag ($tpl) ---"

    server_arg=()
    if (( SERVED == 1 )); then
        server_arg+=(--no-server)
    fi

    python3 "$OMK/eval/omk_eval.py" \
        --model "$MODEL" \
        --template "$tpl" \
        --backend vllm \
        --quant "$QUANT" \
        --port "$PORT" \
        --served-name "$NAME" \
        --results-dir "$RESULTS" \
        "${server_arg[@]}" \
        2>&1 | tee -a "$LOGS/suite_${NAME}_${tag}.log"
    rc=$?
    if (( rc != 0 )); then
        log "FAIL $tag (rc=$rc) — continuing with next bench"
    else
        log "OK   $tag"
    fi
    SERVED=1
done

# Tear down the shared server explicitly — omk_eval --no-server skips
# shutdown, so we own the final pkill.
log "tearing down vllm server on port $PORT"
pkill -f "vllm.*--port $PORT" 2>/dev/null || true
sleep 3

log "=== suite done ==="
log "Per-bench summaries:"
for tag in "${TAGS[@]}"; do
    tpl="${TEMPLATE[$tag]:-}"
    f="$RESULTS/$tpl/summary.json"
    [[ -f "$f" ]] && log "  $tag -> $f" || log "  $tag -> MISSING"
done
