#!/bin/bash
# longer_smoke_v5coder.sh — same triad as longer_smoke_t18_candidates.sh,
# applied to v5-coder candidate(s) so they slot into the comparison table
# alongside v4 baseline + B2/B4/D1/B3.

set -euo pipefail
cd /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models

CANDIDATES=(
    "C1_max_codetb"
)
# Rebalanced 2026-05-17 (smoke for T18 qualification, not full validation):
# - gsm8k_30 — math sanity, confirms math hasn't collapsed
# - humaneval_30 — stride from HE-164, v4/128e anchors precomputed at 29/30
# - lcb_medium_15 — first 15 by date, v4 anchor 12/15, 128e 13/15
# Total wall ~1h30m-2h. Anchor scores extracted from existing v4 + 128e
# samples files; no fresh baseline runs needed.
TEMPLATES=("gsm8k_30" "humaneval_30" "lcb_medium_15")

SRC_HF="google/gemma-4-26B-A4B-it"
RESULTS="eval_results_vllm_suite/v5fixed_t18_longer_smoke/v5coder"
PORT=8195
LOGS="logs/v5coder_longer_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS" "$RESULTS"

for TAG in "${CANDIDATES[@]}"; do
    NVFP="google/gemma-4-A4B-98e-v5coder-${TAG}-NVFP4A16"
    [ -d "$NVFP" ] || { echo "[FATAL] $NVFP not found — build first"; exit 1; }
done
echo "[preflight] $(echo "${CANDIDATES[@]}" | wc -w) candidate(s) + 3 templates present — OK"

eval_one() {
    local TAG=$1 TPL=$2
    local NVFP="google/gemma-4-A4B-98e-v5coder-${TAG}-NVFP4A16"
    local SERVED="98e_v5coder_${TAG}_nvfp4a16"
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

for TAG in "${CANDIDATES[@]}"; do
    echo; echo "=== candidate: $TAG ==="
    for TPL in "${TEMPLATES[@]}"; do
        eval_one "$TAG" "$TPL"
    done
done

# Summary
parse_lm_eval() {
    local TAG=$1 TPL=$2 METRIC=$3
    local SERVED="98e_v5coder_${TAG}_nvfp4a16"
    local R_DIR="$RESULTS/$TAG/$TPL/$TPL/$SERVED/lm_eval_out/$SERVED"
    local R=$(ls -t "$R_DIR"/results_*.json 2>/dev/null | head -1 || true)
    [ -z "$R" ] && { echo "?"; return; }
    /root/anaconda3/envs/omnimergekit/bin/python -c "
import json
r=json.load(open('$R'))['results']
v=list(r.values())[0]
# try several metric key variants (lm-eval reports HE as pass@1,extract_chat)
for k in ['$METRIC', '$METRIC,extract_chat', '$METRIC,strict-match', '$METRIC,flexible-extract']:
    if k in v:
        print(v[k]); break
else:
    print('?')
"
}
parse_lcb() {
    local TAG=$1
    local SERVED="98e_v5coder_${TAG}_nvfp4a16"
    local R=$(ls -t "$RESULTS/$TAG/lcb_medium_5/lcb_medium_5/$SERVED"/lcb_result*.json 2>/dev/null | head -1 || true)
    [ -z "$R" ] && { echo "?"; return; }
    /root/anaconda3/envs/omnimergekit/bin/python -c "
import json
r=json.load(open('$R'))
print(r.get('pass_at_1', r.get('pass@1', '?')))
"
}

echo
echo "===== v5-coder longer-smoke summary ====="
printf "%-22s | %8s | %8s | %8s | %8s\n" "candidate" "gsm_str" "gsm_flx" "HE-20" "LCB-5"
printf "%-22s-+-%8s-+-%8s-+-%8s-+-%8s\n" "----------------------" "--------" "--------" "--------" "--------"
for TAG in "${CANDIDATES[@]}"; do
    GS=$(parse_lm_eval "$TAG" "gsm8k_v128pass" "exact_match")
    GF=$(parse_lm_eval "$TAG" "gsm8k_v128pass" "exact_match")
    HE=$(parse_lm_eval "$TAG" "humaneval_20" "pass@1")
    LC=$(parse_lcb "$TAG")
    printf "%-22s | %8s | %8s | %8s | %8s\n" "$TAG" "$GS" "$GF" "$HE" "$LC"
done | tee "$LOGS/_summary.tsv"
echo; echo "logs:    $LOGS/"
echo "results: $RESULTS/"
