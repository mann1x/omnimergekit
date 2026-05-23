#!/usr/bin/env bash
# Post-cohort GPQA re-run at thinking_token_budget=24576 for v5-coder ONLY.
#
# Context (2026-05-22):
#   - Template gpqa_diamond_full.yaml was patched: budget 12288 -> 24576,
#     max_gen_toks 16384 -> 32768. This is the new canonical.
#   - v5-coder GPQA already ran on stack@2 at the OLD 12k budget = 59.60%
#     (vs published 68.69%; 22/198 questions truncated mid-reasoning).
#   - v6-coder cohort is still queued (will fire after v5-coder LCB completes)
#     and will INHERIT the patched 24k budget — no separate v6 re-run needed.
#   - Goal: re-run v5-coder GPQA at 24k so v5 vs v6 comparison stays apples-to-apples.
#
# Discipline: wait for the full cohort orchestrator (PID 120030) to exit so the
# GPU is free, then run the v5-coder GPQA re-eval alone.
#
# Logs: /srv/.../backup_models/logs/post_cohort_gpqa24k_*.log

set -euo pipefail

ROOT="/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models"
OMK="/shared/dev/omnimergekit"
COHORT_PID=120030
VARIANT=v5coder
SERVED=98e_v5_coder_nvfp4a16
MODEL_DIR="$ROOT/google/gemma-4-A4B-98e-v5-coder-NVFP4A16"

LOGTS=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT/logs/post_cohort_gpqa24k_${VARIANT}_${LOGTS}.log"
mkdir -p "$ROOT/logs"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Gate 1: wait for cohort orchestrator exit (v5-coder LCB + full v6-coder cohort)
log "Waiting for cohort orchestrator PID $COHORT_PID to exit…"
while kill -0 "$COHORT_PID" 2>/dev/null; do
    sleep 60
done
log "Cohort orchestrator exited — GPU should be free."

# Gate 2: brief settle for vLLM to release the GPU
sleep 30

# Gate 3: confirm patched template (defence in depth — if someone reverted it, abort)
TBUDGET=$(grep -E "^\s*thinking_token_budget" "$OMK/eval/templates/gpqa_diamond_full.yaml" | head -1 | awk '{print $2}')
if [ "$TBUDGET" != "24576" ]; then
    log "ABORT: gpqa_diamond_full.yaml has thinking_token_budget=$TBUDGET, expected 24576. Template was reverted; investigate."
    exit 1
fi
log "Template budget confirmed at 24576."

# Gate 4: purge v5-coder GPQA sqlite cache + result files (force re-sample)
CACHE_DIR="$ROOT/eval_results_vllm_suite/$VARIANT/gpqa_diamond_full/$SERVED/sqlite_cache"
OLD_RESULTS="$ROOT/eval_results_vllm_suite/$VARIANT/gpqa_diamond_full/$SERVED/lm_eval_out"
ARCHIVE="$ROOT/eval_results_vllm_suite_archive_12k/$VARIANT/gpqa_diamond_full"
mkdir -p "$ARCHIVE"
if [ -d "$OLD_RESULTS" ]; then
    log "Archiving 12k results → $ARCHIVE"
    cp -a "$OLD_RESULTS" "$ARCHIVE/lm_eval_out_12k_${LOGTS}"
    cp -a "$CACHE_DIR" "$ARCHIVE/sqlite_cache_12k_${LOGTS}" 2>/dev/null || true
fi
log "Purging cache + results for fresh 24k re-sample"
rm -rf "$CACHE_DIR" "$OLD_RESULTS"

# Gate 5: fire the GPQA-only re-eval via canonical omk_eval
log "Launching v5-coder GPQA re-eval at 24k budget…"
cd "$OMK"
OUTDIR="$ROOT/eval_results_vllm_suite/$VARIANT"
/root/anaconda3/envs/omnimergekit/bin/python eval/omk_eval.py \
    --model "$MODEL_DIR" \
    --served-name "$SERVED" \
    --template gpqa_diamond_full \
    --backend vllm \
    --output-dir "$OUTDIR" \
    2>&1 | tee -a "$LOG"

# Gate 6: summarize before vs after
log "=== GPQA: v5-coder 12k vs 24k ==="
OLD_RES=$(find "$ARCHIVE" -name "results_*.json" 2>/dev/null | sort | tail -1)
NEW_RES=$(find "$ROOT/eval_results_vllm_suite/$VARIANT/gpqa_diamond_full" -name "results_*.json" 2>/dev/null | sort | tail -1)
python3 - <<PY | tee -a "$LOG"
import json
def score(p):
    if not p: return None
    r = json.load(open(p))
    for _, m in r.get("results", {}).items():
        if "exact_match,flexible-extract" in m: return m["exact_match,flexible-extract"]
    return None
o = score("$OLD_RES")
n = score("$NEW_RES")
if o is not None and n is not None:
    print(f"v5-coder GPQA  12k={o*100:.2f}%  24k={n*100:.2f}%  delta={(n-o)*100:+.2f}pp")
else:
    print(f"MISSING: 12k={o} 24k={n}  paths: old=$OLD_RES new=$NEW_RES")
PY

log "post_cohort_gpqa_24k_rerun.sh complete."
