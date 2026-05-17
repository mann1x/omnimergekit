#!/bin/bash
# longer_smoke_v4_baseline.sh — run the longer-smoke triad on the
# 98e-v4-cd-max NVFP4A16 baseline so the four v5fixed-sweep candidates
# can be compared against a real anchor (not just the definitional
# 128e=100% on filtered sets).
#
# Mirror of longer_smoke_t18_candidates.sh shape; same templates
# (gsm8k_v128pass + humaneval_20 + lcb_medium_5), same backend (vLLM
# NVFP4A16 on port 8195), same results-dir layout. Launch AFTER the
# 4-candidate run lands so the GPU is free.

set -euo pipefail
cd /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models

NVFP="google/Gemma-4-A4B-98e-v4-NVFP4A16"
SERVED="98e_v4_t18_baseline_nvfp4a16"
SRC_HF="google/gemma-4-26B-A4B-it"
RESULTS="eval_results_vllm_suite/v5fixed_t18_longer_smoke/v4_baseline"
PORT=8195
LOGS="logs/v5fixed_v4_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS" "$RESULTS"

TEMPLATES=("gsm8k_v128pass" "humaneval_20" "lcb_medium_5")

if [ ! -d "$NVFP" ]; then
    echo "[FATAL] $NVFP not found"; exit 1
fi
echo "[preflight] $NVFP present — OK"

eval_one() {
    local TPL=$1
    local OUTDIR="$RESULTS/$TPL"
    if ls "$OUTDIR/$TPL/$SERVED/lm_eval_out/$SERVED"/results_*.json >/dev/null 2>&1 || \
       ls "$OUTDIR/$TPL/$SERVED"/results_*.json >/dev/null 2>&1 || \
       ls "$OUTDIR/$TPL/$SERVED"/lcb_result*.json >/dev/null 2>&1; then
        echo "[v4_baseline/$TPL] results exist, skip"; return 0
    fi
    echo "[v4_baseline/$TPL] $(date +%H:%M:%S) start"
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
        2>&1 | tee -a "$LOGS/v4_baseline_${TPL}.log"
    echo "[v4_baseline/$TPL] $(date +%H:%M:%S) done"
}

for TPL in "${TEMPLATES[@]}"; do
    eval_one "$TPL"
done

# --- Summary ---
parse_lm_eval() {
    local TPL=$1 METRIC=$2
    local R_DIR="$RESULTS/$TPL/$TPL/$SERVED/lm_eval_out/$SERVED"
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
    local R=$(ls -t "$RESULTS/lcb_medium_5/lcb_medium_5/$SERVED"/lcb_result*.json 2>/dev/null | head -1 || true)
    [ -z "$R" ] && { echo "?"; return; }
    /root/anaconda3/envs/omnimergekit/bin/python -c "
import json
r=json.load(open('$R'))
print(r.get('pass_at_1', r.get('pass@1', '?')))
"
}

echo
echo "===== v4 baseline longer-smoke summary ====="
GS=$(parse_lm_eval "gsm8k_v128pass" "exact_match,strict-match")
GF=$(parse_lm_eval "gsm8k_v128pass" "exact_match,flexible-extract")
HE=$(parse_lm_eval "humaneval_20" "pass@1")
LC=$(parse_lcb)
printf "%-22s | %8s | %8s | %8s | %8s\n" "v4_baseline" "$GS" "$GF" "$HE" "$LC" | tee "$LOGS/_summary.tsv"
echo
echo "logs:    $LOGS/"
echo "results: $RESULTS/"
