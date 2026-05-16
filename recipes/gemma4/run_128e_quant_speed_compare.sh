#!/usr/bin/env bash
# 128e Gemma 4 26B — compare vLLM 4-bit quant runtimes, then full LCB-55
# on the faster ones.
#
# Pipeline:
#   Stage 0  — (post-validation, manual) confirm 128e_nvfp4a16 LCB-55
#              result is good, then run this script.
#   Stage 1  — build the missing 4-bit quants in vLLM-compatible formats:
#                AWQ-4bit  (autoawq 0.2.9, in vllm env)
#                GPTQ-4bit (gptqmodel 7.0, in modelopt env)
#                NVFP4A16  (already built — skip)
#              Each goes to $WS/google/Gemma-4-26B-A4B-it-<METHOD>.
#   Stage 2  — speed smoke (3 prompts × 4096 max_tokens) per quant.
#              Records median tok/s into $RESULTS/speed_smoke.json.
#   Stage 3  — for each quant with median_tok/s >= NVFP4A16_REF, run
#              full LCB-55. NVFP4A16_REF is read from the existing run.
#              Skip the others — they're not worth ~2h each.
#
# After Stage 3, eval/omk_summarize.py rolls the per-quant
# summary.json files into a card-ready table.
#
# Usage:
#   bash run_128e_quant_speed_compare.sh           # all stages
#   bash run_128e_quant_speed_compare.sh stage1    # just build quants
#   bash run_128e_quant_speed_compare.sh stage2    # just speed smoke
#   bash run_128e_quant_speed_compare.sh stage3    # just LCB-55 on fast ones
#
# Stages 1+2 are safe to re-run (idempotent). Stage 3 reuses any cached
# samples.jsonl from previous attempts via the LCB shim's resume logic.
set -uo pipefail

STAGE="${1:-all}"

WS="${WS:-/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models}"
OMK="${OMK:-/shared/dev/omnimergekit}"
RESULTS="${RESULTS:-$WS/eval_results_quant_compare/128e}"
LOGS="${LOGS:-$WS/logs}"
PORT="${PORT:-8195}"

BF16="$WS/google/gemma-4-26B-A4B-it"
TOK="$BF16"

declare -A QUANT_DST
# Already validated NVFP4A16 (modelopt) → reference.
QUANT_DST[nvfp4a16]="$WS/google/Gemma-4-26B-A4B-it-NVFP4A16"
# INT4-AWQ via modelopt — currently broken on Gemma 4 (CUDA index assert
# in AWQ calib path). Kept here for documentation; not in active loop.
QUANT_DST[int4_awq]="$WS/google/Gemma-4-26B-A4B-it-INT4-AWQ"
# MXFP4 via modelopt (Microsoft/OCP FP4, group=32). Tensor-level path
# like NVFP4A16, no AWQ smoothing → should avoid the int4_awq crash.
QUANT_DST[mxfp4]="$WS/google/Gemma-4-26B-A4B-it-MXFP4"
# NVFP4 + AWQ-Lite smoothing via modelopt. Same NVFP4 weight format as
# NVFP4A16 but with lite-mode AWQ scaling for better quality.
QUANT_DST[nvfp4_awq_lite]="$WS/google/Gemma-4-26B-A4B-it-NVFP4-AWQ-Lite"
# GPTQ-4bit (gptqmodel 7.0). NOTE: does NOT compress Gemma 4 fused MoE
# experts — output is BF16-sized. Kept as a row for documentation; the
# stage1 loop SKIPS it (see METHOD list below).
QUANT_DST[gptq]="$WS/google/Gemma-4-26B-A4B-it-GPTQ4"
# llm-compressor (vllm-project/llm-compressor) quants — replaces autoawq
# (which does not support Gemma 4). Three flavours per user request:
#   W4A16   = INT4 weights + FP16 activations (AWQ replacement)
#   W4A8    = INT4 weights + INT8 activations (faster, slightly lower quality)
#   NVFP4   = FP4 weights AND FP4 activations (full FP4 end-to-end)
QUANT_DST[w4a16_lc]="$WS/google/Gemma-4-26B-A4B-it-W4A16-LC"
QUANT_DST[w4a8_lc]="$WS/google/Gemma-4-26B-A4B-it-W4A8-LC"
QUANT_DST[nvfp4_lc]="$WS/google/Gemma-4-26B-A4B-it-NVFP4-LC"
# autoawq path is dead-end for Gemma 4 (TypeError: gemma4 isn't supported yet.)
# Kept as a row for documentation, but stage1 will SKIP autoawq below.

mkdir -p "$RESULTS" "$LOGS"
SUITE_LOG="$LOGS/quant_compare_128e.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$SUITE_LOG"; }

# ── stage 1: build missing AWQ + GPTQ quants ──────────────────────────────
stage1() {
    log "=== stage 1: build 4-bit quants ==="
    # autoawq is excluded — does not support gemma4 model_type
    # (2026-05-12, autoawq 0.2.9, last release before deprecation).
    for METHOD in mxfp4 nvfp4_awq_lite nvfp4a16; do
        local dst="${QUANT_DST[$METHOD]}"
        if [[ -f "$dst/config.json" ]]; then
            log "[$METHOD] already at $dst, skipping"
            continue
        fi
        log "[$METHOD] quantizing $BF16 -> $dst"
        # quantize_any.py dispatches to the correct conda env per method.
        python3 "$OMK/scripts/quantize_any.py" \
            --src "$BF16" --dst "$dst" --method "$METHOD" \
            2>&1 | tee -a "$LOGS/quant_${METHOD}.log"
        if [[ ! -f "$dst/config.json" ]]; then
            log "[$METHOD] QUANT FAILED — see $LOGS/quant_${METHOD}.log"
        else
            log "[$METHOD] done: $(du -sh "$dst" | cut -f1)"
        fi
    done
}

# ── stage 2: serve each quant briefly, run the 3-prompt speed smoke ───────
stage2() {
    log "=== stage 2: speed smoke (median tok/s per quant) ==="
    local smoke_json="$RESULTS/speed_smoke.json"
    : > "$smoke_json.tmp"
    for METHOD in nvfp4a16 mxfp4 nvfp4_awq_lite; do
        local dst="${QUANT_DST[$METHOD]}"
        if [[ ! -f "$dst/config.json" ]]; then
            log "[$METHOD] not built — skipping smoke"
            continue
        fi
        local name="128e_$METHOD"
        log "[$METHOD] launching vllm on port $PORT ..."
        pkill -f "vllm.*--port $PORT" 2>/dev/null || true; sleep 4
        local slog="$LOGS/vllm_smoke_${METHOD}.log"
        nohup /root/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
            --model "$dst" --served-model-name "$name" \
            --port "$PORT" --gpu-memory-utilization 0.92 \
            --max-model-len 8192 --max-num-batched-tokens 4096 \
            --enforce-eager --dtype bfloat16 --trust-remote-code \
            > "$slog" 2>&1 &
        disown
        # wait for ready
        for i in $(seq 1 90); do
            curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
            sleep 4
        done
        if ! curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
            log "[$METHOD] vllm failed to start — see $slog"
            pkill -f "vllm.*--port $PORT" 2>/dev/null || true
            continue
        fi
        # warmup
        curl -sf -X POST "http://localhost:$PORT/v1/chat/completions" \
            -H "content-type: application/json" \
            -d "{\"model\":\"$name\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}" \
            >/dev/null 2>&1
        log "[$METHOD] running smoke ..."
        /root/anaconda3/envs/omnimergekit/bin/python "$OMK/eval/quant_speed_smoke.py" \
            --url "http://localhost:$PORT" --name "$name" \
            --max-tokens 4096 --n 3 --json \
            2>>"$LOGS/quant_smoke_${METHOD}.log" >> "$smoke_json.tmp" || \
            log "[$METHOD] smoke failed"
        pkill -f "vllm.*--port $PORT" 2>/dev/null || true; sleep 4
    done
    mv "$smoke_json.tmp" "$smoke_json"
    log "smoke summary:"
    /root/anaconda3/envs/omnimergekit/bin/python -c "
import json, sys
for line in open('$smoke_json'):
    d = json.loads(line)
    if d.get('median_tok_per_s'):
        print(f\"  {d['name']:30s} median={d['median_tok_per_s']:6.2f} tok/s  mean={d['mean_tok_per_s']:6.2f}\")
    else:
        print(f\"  {d['name']:30s} ERROR  {d.get('errors')}\")
" | tee -a "$SUITE_LOG"
}

# ── stage 3: full LCB-55 on quants matching or beating NVFP4A16 ───────────
stage3() {
    log "=== stage 3: full LCB-55 on faster quants ==="
    local smoke_json="$RESULTS/speed_smoke.json"
    [[ -f "$smoke_json" ]] || { log "no smoke results at $smoke_json — run stage2 first"; return 1; }
    # parse smoke results — qualifying = median_tok/s >= 0.95 × nvfp4a16's
    local plan
    plan=$(/root/anaconda3/envs/omnimergekit/bin/python -c "
import json
rows = [json.loads(l) for l in open('$smoke_json')]
ref = {r['name']: r.get('median_tok_per_s') or 0 for r in rows}
nvfp = ref.get('128e_nvfp4a16', 0)
print(f'# nvfp4a16 reference: {nvfp:.2f} tok/s')
for name, v in ref.items():
    if v >= nvfp * 0.95:
        print(name, f'{v:.2f}')
" )
    echo "$plan" | tee -a "$SUITE_LOG"
    echo "$plan" | awk '/^[^#]/ {print $1}' | while read name; do
        [[ -z "$name" ]] && continue
        # map name back to dir
        local method="${name#128e_}"
        local dst="${QUANT_DST[$method]:-}"
        [[ -d "$dst" ]] || { log "[$name] no dir $dst — skip"; continue; }
        log "[$name] full LCB-55 against $dst"
        pkill -f "vllm.*--port $PORT" 2>/dev/null || true; sleep 4
        # launch vllm for the full LCB-55 run
        nohup /root/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
            --model "$dst" --served-model-name "$name" \
            --port "$PORT" --gpu-memory-utilization 0.92 \
            --max-model-len 32768 --max-num-batched-tokens 4096 \
            --enforce-eager --dtype bfloat16 --trust-remote-code \
            > "$LOGS/vllm_lcb_${method}.log" 2>&1 &
        disown
        for i in $(seq 1 120); do
            curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
            sleep 5
        done
        # warmup
        curl -sf -X POST "http://localhost:$PORT/v1/chat/completions" \
            -H "content-type: application/json" \
            -d "{\"model\":\"$name\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}" \
            >/dev/null 2>&1
        local outdir="$RESULTS/lcb_med_55_$method"
        mkdir -p "$outdir"
        /root/anaconda3/envs/omnimergekit/bin/python "$OMK/eval/lcb/lcb_llama_server.py" \
            --name "$name" --base-url "http://localhost:$PORT" \
            --max-tokens 16384 --http-timeout 1200 \
            --difficulty medium --min-date 2024-10-01 \
            --limit 999 \
            --output "$outdir/lcb.json" \
            2>&1 | tee -a "$LOGS/lcb_${method}.log"
        pkill -f "vllm.*--port $PORT" 2>/dev/null || true; sleep 4
    done
}

case "$STAGE" in
    all)    stage1; stage2; stage3 ;;
    stage1) stage1 ;;
    stage2) stage2 ;;
    stage3) stage3 ;;
    *)      log "unknown stage '$STAGE'"; exit 2 ;;
esac

log "=== done ==="
