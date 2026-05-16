#!/usr/bin/env bash
# 3-way LCB-medium-55 on Q6_K llama.cpp — the fast counterpart to
# run_a4b_nvfp4_lcb55_3way.sh.
#
# Why two recipes:
#   - NVFP4A16 sustains ~22 tok/s on a 3090 (Marlin kernels, enforce-eager).
#   - llama.cpp Q6_K sustains 60-80 tok/s (cuBLAS path, parallel=2).
#   - LCB-medium emits ~14k generation tokens / problem; 55 problems ×
#     3× delta = ~1.5 wall-hour cost per variant.
#
# Use this recipe for production runs. Use the NVFP4 recipe only when
# explicitly comparing the two quant paths.
#
# All three Q6_K GGUFs already exist locally:
#   $WS/google/gemma-4-26B-A4B-it-Q6_K.gguf       (128e)
#   $WS/google/gemma-4-A4B-98e-v3-Q6_K.gguf       (v3)
#   $WS/google/gemma-4-A4B-98e-v4-Q6_K.gguf       (v4 — built if missing)
#
# omk_eval.py launch_llama auto-applies the bench-typed flags for
# LCB tasks: `--jinja --reasoning off`. We override nothing here.
#
# Usage:
#   bash run_a4b_q6k_lcb55_3way.sh           # all three
#   bash run_a4b_q6k_lcb55_3way.sh v3        # just 98e v3
#   bash run_a4b_q6k_lcb55_3way.sh v4 128e
set -uo pipefail

WS="${WS:-/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models}"
OMK="${OMK:-/shared/dev/omnimergekit}"
RESULTS="${RESULTS:-$WS/eval_results_q6k/lcb_med_55}"
LOGS="${LOGS:-$WS/logs}"
PORT="${PORT:-8099}"
LLAMA="${LLAMA:-/opt/llama.cpp}"

mkdir -p "$RESULTS" "$LOGS"

declare -A GGUF NAME BF16
GGUF[v3]="$WS/google/gemma-4-A4B-98e-v3-Q6_K.gguf"
NAME[v3]="98e_v3_q6k"
BF16[v3]="$WS/google/gemma-4-A4B-98e-v3-it"

GGUF[v4]="$WS/google/gemma-4-A4B-98e-v4-Q6_K.gguf"
NAME[v4]="98e_v4_q6k"
BF16[v4]="$WS/google/gemma-4-A4B-98e-v4-it"

GGUF[128e]="$WS/google/gemma-4-26B-A4B-it-Q6_K.gguf"
NAME[128e]="128e_q6k"
BF16[128e]="$WS/google/gemma-4-26B-A4B-it"

VARIANTS=("${@:-v3 v4 128e}")
[[ "${VARIANTS[*]}" == "v3 v4 128e" ]] && VARIANTS=(v3 v4 128e)

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOGS/lcb55_q6k_3way.log"; }

ensure_q6k() {
    local v="$1"
    local out="${GGUF[$v]}"
    local src="${BF16[$v]}"
    [[ -f "$out" ]] && return 0
    [[ -f "$src/config.json" ]] || { log "ABORT: no BF16 source at $src"; return 1; }
    local f16="${out%.Q6_K.gguf}.F16.gguf"
    log "[$v] convert BF16 -> F16: $src -> $f16"
    python3 "$LLAMA/convert_hf_to_gguf.py" "$src" --outfile "$f16" --outtype f16 \
        2>&1 | tee -a "$LOGS/nvfp4_quant_${v}.log" | tail -3
    [[ -f "$f16" ]] || { log "[$v] F16 convert FAILED"; return 1; }
    log "[$v] quantize F16 -> Q6_K: $f16 -> $out"
    "$LLAMA/build/bin/llama-quantize" "$f16" "$out" Q6_K \
        2>&1 | tee -a "$LOGS/nvfp4_quant_${v}.log" | tail -3
    rm -f "$f16"
    [[ -f "$out" ]]
}

for v in "${VARIANTS[@]}"; do
    if [[ -z "${GGUF[$v]:-}" ]]; then
        log "ERROR: unknown variant '$v' — pick v3, v4, or 128e"
        continue
    fi
    log "=== variant: $v ==="
    if ! ensure_q6k "$v"; then
        log "[$v] Q6_K missing and rebuild failed — skipping"
        continue
    fi

    log "[$v] LCB-55 via omk_eval (llama.cpp Q6_K)"
    # tokenizer must be the original 128e dir for v3/v4 (chat template + bpe).
    python3 "$OMK/eval/omk_eval.py" \
        --model "${GGUF[$v]}" \
        --template lcb_medium_55 \
        --backend llama \
        --port "$PORT" \
        --results-dir "$RESULTS" \
        --served-name "${NAME[$v]}" \
        --tokenizer "${BF16[128e]}" \
        2>&1 | tee -a "$LOGS/lcb55_${v}_q6k_omk.log"

    log "[$v] done. Result at $RESULTS/lcb_med_55/${NAME[$v]}/"
done

log "=== all variants processed ==="
for v in "${VARIANTS[@]}"; do
    [[ -n "${NAME[$v]:-}" ]] && log "  $v -> $RESULTS/lcb_med_55/${NAME[$v]}/"
done
