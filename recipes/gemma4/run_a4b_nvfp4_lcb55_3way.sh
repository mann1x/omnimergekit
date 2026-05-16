#!/usr/bin/env bash
# 3-way LCB-medium-55 comparison: 98e-v3 vs 98e-v4 vs 128e, all at
# NVFP4A16 via modelopt, all scored with the patched lcb_helpers shim
# (validated 13/13 against lcb_runner.evaluation.testing_util on 2026-05-12).
#
# Quant path: BF16 dir → NVFP4A16 via modelopt → vLLM /v1/chat/completions.
# Eval path: omk_eval.py --template lcb_medium_55 --backend vllm.
#
# Sequential by design: 24 GB VRAM doesn't fit two 26B models at once,
# and we want apples-to-apples timing for the comparison anyway.
#
# Usage:
#   bash run_a4b_nvfp4_lcb55_3way.sh           # all three
#   bash run_a4b_nvfp4_lcb55_3way.sh v3        # just 98e_v3
#   bash run_a4b_nvfp4_lcb55_3way.sh v4 128e   # v4 and 128e
set -uo pipefail

WS="${WS:-/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models}"
OMK="${OMK:-/shared/dev/omnimergekit}"
RESULTS="${RESULTS:-$WS/eval_results_vllm4bit/lcb_med_55q}"
LOGS="${LOGS:-$WS/logs}"
PORT="${PORT:-8195}"

mkdir -p "$RESULTS" "$LOGS"

# Variant map: tag → (bf16-source-dir, nvfp4-output-dir, served-name)
declare -A SRC OUT NAME
SRC[v3]="$WS/google/gemma-4-A4B-98e-v3-it"
OUT[v3]="$WS/google/gemma-4-A4B-98e-v3-it-NVFP4A16"
NAME[v3]="98e_v3_nvfp4a16"

SRC[v4]="$WS/google/gemma-4-A4B-98e-v4-it"
OUT[v4]="$WS/google/gemma-4-A4B-98e-v4-it-NVFP4A16"
NAME[v4]="98e_v4_nvfp4a16"

SRC[128e]="$WS/google/Gemma-4-26B-A4B-it"  # full BF16 — only used if NVFP4 missing
OUT[128e]="$WS/google/Gemma-4-26B-A4B-it-NVFP4A16"
NAME[128e]="128e_nvfp4a16"

# Variants to process (default: all)
VARIANTS=("${@:-v3 v4 128e}")
[[ "${VARIANTS[*]}" == "v3 v4 128e" ]] && VARIANTS=(v3 v4 128e)

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOGS/lcb55_3way.log"; }

for v in "${VARIANTS[@]}"; do
    if [[ -z "${SRC[$v]:-}" ]]; then
        log "ERROR: unknown variant '$v' — pick v3, v4, or 128e"
        continue
    fi
    log "=== variant: $v ==="
    src="${SRC[$v]}"
    dst="${OUT[$v]}"
    name="${NAME[$v]}"

    # 1. Quantize if NVFP4A16 dir missing
    if [[ ! -f "$dst/config.json" ]]; then
        if [[ ! -f "$src/config.json" ]]; then
            log "ABORT: no BF16 source at $src and no NVFP4A16 at $dst"
            continue
        fi
        log "[$v] quantizing $src -> $dst (NVFP4A16 via modelopt)"
        python3 "$OMK/scripts/quantize_any.py" \
            --src "$src" --dst "$dst" --method nvfp4a16 \
            2>&1 | tee -a "$LOGS/nvfp4_quant_$v.log"
        if [[ ! -f "$dst/config.json" ]]; then
            log "[$v] QUANT FAILED — see $LOGS/nvfp4_quant_$v.log"
            continue
        fi
        log "[$v] quant done: $(du -sh "$dst" | cut -f1)"
    else
        log "[$v] NVFP4A16 exists at $dst, skipping quant"
    fi

    # 2. Eval via omk_eval
    log "[$v] LCB-55 via omk_eval"
    python3 "$OMK/eval/omk_eval.py" \
        --model "$dst" \
        --template lcb_medium_55 \
        --backend vllm \
        --quant auto \
        --port "$PORT" \
        --results-dir "$RESULTS" \
        --served-name "$name" \
        2>&1 | tee -a "$LOGS/lcb55_${v}_omk.log"

    log "[$v] eval done. Summary at $RESULTS/lcb_med_55/$name/summary.json"
done

log "=== all variants processed ==="
log "Results dirs:"
for v in "${VARIANTS[@]}"; do
    [[ -n "${NAME[$v]:-}" ]] && log "  $v -> $RESULTS/lcb_med_55/${NAME[$v]}/"
done
