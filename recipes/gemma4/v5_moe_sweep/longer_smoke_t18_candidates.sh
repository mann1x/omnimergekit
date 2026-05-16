#!/bin/bash
# longer_smoke_t18_candidates.sh — longer-smoke validation for T18 candidates.
#
# Runs three benches (gsm8k_v128pass, humaneval_20, lcb_medium_5) on the
# four sweep candidates that emerged from T19:
#   B4_max_broad     — all-rounder (only variant positive on all 4 smoke axes)
#   B2_max_mathcode  — math-flex king (top of gsm_flex column)
#   D1_lp4_mathcode  — math-strict pole (top of gsm_strict column)
#   B3_max_tgtheavy  — code-pure pole (rescored HE 1.0 + LCB 1.0)
#
# Selection criterion for every bench: every problem MUST be a verified
# 128e-NVFP4A16 PASS on the current vLLM recipe (parser=gemma4 +
# thinking_budget=12288). A variant FAIL is then signal (pruning damage),
# not noise (universally-hard problem).
#
# Idempotent: skip a (TAG, TEMPLATE) cell if a results_*.json already exists.
#
# Total: 4 variants × 3 benches = 12 omk_eval calls. Estimated wall ~6-8 h
# on a single 3090 (vLLM NVFP4A16 throughput).

set -euo pipefail
cd /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models

SRC_HF="google/gemma-4-26B-A4B-it"
RESULTS="eval_results_vllm_suite/v5fixed_t18_longer_smoke"
PORT=8195
LOGS="logs/v5fixed_longer_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS" "$RESULTS"

CANDIDATES=(
    "B4_max_broad"
    "B2_max_mathcode"
    "D1_lp4_mathcode"
    "B3_max_tgtheavy"
)
TEMPLATES=(
    "gsm8k_v128pass"
    "humaneval_20"
    "lcb_medium_5"
)

# --- Preflight: variant dirs + templates exist ---
for TAG in "${CANDIDATES[@]}"; do
    NVFP="google/gemma-4-A4B-98e-v5fixed-sweep-${TAG}-NVFP4A16"
    if [ ! -d "$NVFP" ]; then
        echo "[FATAL] $NVFP not found"; exit 1
    fi
done
for TPL in "${TEMPLATES[@]}"; do
    if [ ! -f /shared/dev/omnimergekit/eval/templates/${TPL}.yaml ]; then
        echo "[FATAL] template ${TPL}.yaml not found"; exit 1
    fi
done
echo "[preflight] 4 candidates + 3 templates present — OK"

# --- Helper ---
eval_one() {
    local TAG=$1 TPL=$2
    local NVFP="google/gemma-4-A4B-98e-v5fixed-sweep-${TAG}-NVFP4A16"
    local SERVED="98e_v5fixed_t18_${TAG}_nvfp4a16"
    local OUTDIR="$RESULTS/$TAG/$TPL"
    if ls "$OUTDIR/$TPL/$SERVED/lm_eval_out/$SERVED"/results_*.json >/dev/null 2>&1 || \
       ls "$OUTDIR/$TPL/$SERVED"/results_*.json >/dev/null 2>&1 || \
       ls "$OUTDIR/$TPL/$SERVED"/lcb_result*.json >/dev/null 2>&1; then
        echo "[$TAG/$TPL] results exist, skip"; return 0
    fi
    echo "[$TAG/$TPL] $(date +%H:%M:%S) start"
    PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH PYTHONDONTWRITEBYTECODE=1 \
    VLLM_PYTHON=/root/anaconda3/envs/vllm/bin/python \
    /root/anaconda3/envs/omnimergekit/bin/python /shared/dev/omnimergekit/eval/omk_eval.py \
        --model "$NVFP" \
        --template "$TPL" \
        --backend vllm \
        --port "$PORT" \
        --served-name "$SERVED" \
        --tokenizer "$SRC_HF" \
        --max-model-len 40960 \
        --results-dir "$OUTDIR" \
        2>&1 | tee -a "$LOGS/${TAG}_${TPL}.log"
    echo "[$TAG/$TPL] $(date +%H:%M:%S) done"
}

# --- Main loop ---
for TAG in "${CANDIDATES[@]}"; do
    echo
    echo "=== candidate: $TAG ==="
    for TPL in "${TEMPLATES[@]}"; do
        eval_one "$TAG" "$TPL"
    done
done

# --- Summary parsing ---
parse_lm_eval() {
    local TAG=$1 TPL=$2 METRIC=$3
    local SERVED="98e_v5fixed_t18_${TAG}_nvfp4a16"
    local R_DIR="$RESULTS/$TAG/$TPL/$TPL/$SERVED/lm_eval_out/$SERVED"
    local R=$(ls -t "$R_DIR"/results_*.json 2>/dev/null | head -1 || true)
    [ -z "$R" ] && { echo "?"; return; }
    /root/anaconda3/envs/omnimergekit/bin/python -c "
import json
r=json.load(open('$R'))['results']
v=list(r.values())[0]
print(v.get('$METRIC', '?'))
"
}

parse_lcb() {
    local TAG=$1
    local SERVED="98e_v5fixed_t18_${TAG}_nvfp4a16"
    local R=$(ls -t "$RESULTS/$TAG/lcb_medium_5/lcb_medium_5/$SERVED"/lcb_result*.json 2>/dev/null | head -1 || true)
    [ -z "$R" ] && { echo "?"; return; }
    /root/anaconda3/envs/omnimergekit/bin/python -c "
import json
r=json.load(open('$R'))
print(r.get('pass_at_1', r.get('pass@1', '?')))
"
}

echo
echo "===== T18 candidate longer-smoke summary ====="
printf "%-22s | %8s | %8s | %8s | %8s\n" "candidate" "gsm_str" "gsm_flx" "HE-20" "LCB-5"
printf "%-22s-+-%8s-+-%8s-+-%8s-+-%8s\n" "----------------------" "--------" "--------" "--------" "--------"
for TAG in "${CANDIDATES[@]}"; do
    GS=$(parse_lm_eval "$TAG" "gsm8k_v128pass" "exact_match,strict-match")
    GF=$(parse_lm_eval "$TAG" "gsm8k_v128pass" "exact_match,flexible-extract")
    HE=$(parse_lm_eval "$TAG" "humaneval_20" "pass@1")
    LC=$(parse_lcb "$TAG")
    printf "%-22s | %8s | %8s | %8s | %8s\n" "$TAG" "$GS" "$GF" "$HE" "$LC"
done | tee "$LOGS/_summary.tsv"
echo
echo "logs:    $LOGS/"
echo "results: $RESULTS/"
