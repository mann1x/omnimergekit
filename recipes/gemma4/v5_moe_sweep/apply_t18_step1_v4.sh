#!/bin/bash
# apply_t18_step1_v4.sh — apply T18 Step 1 (free knobs) directly to v4
# NVFP4A16 and re-smoke with the C2-comparable triad (gsm8k_30 +
# humanevalplus_30 + lcb_medium_15) so the results sit alongside the
# v4 anchor / B2 / B4 / D1 / B3 / C2 column in the comparison table.
#
# Pipeline (all idempotent via each script's --restore flag):
#   1. router_topk_dial.py        — set top-k to N
#   2. router_shared_upweight.py  — scale shared mlp by α_shared
#   3. router_soft_transfer.py    — soft-redistribute dropped router rows
#   4. omk_eval                    — gsm8k_30 + humanevalplus_30 + lcb_medium_15
#
# Usage:
#   bash apply_t18_step1_v4.sh [<topk>] [<alpha_shared>] [<alpha_transfer>]
#     defaults: topk=6 alpha_shared=1.2 alpha_transfer=0.3
#
# Restore baseline before another config:
#   bash apply_t18_step1_v4.sh --restore
#
# Result dirs: eval_results_vllm_suite/v4_t18/t<K>_s<aS>_x<aT>/
# Summary:     logs/t18_v4_summary.tsv

set -euo pipefail
cd /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models

NVFP="google/Gemma-4-A4B-98e-v4-NVFP4A16"
DROP_MAP="scripts/cd_multiclass_98e_max_drop_map.json"
BASE_HF="google/gemma-4-26B-A4B-it"
PORT=8194

if [ "${1:-}" = "--restore" ]; then
    echo "[restore] reverting router edits on v4 NVFP"
    /root/anaconda3/envs/omnimergekit/bin/python scripts/router_soft_transfer.py \
        --base-dir "$BASE_HF" --variant-dir "$NVFP" --drop-map "$DROP_MAP" --restore || true
    /root/anaconda3/envs/omnimergekit/bin/python scripts/router_shared_upweight.py \
        --model-dir "$NVFP" --restore || true
    /root/anaconda3/envs/omnimergekit/bin/python scripts/router_topk_dial.py \
        --model-dir "$NVFP" --restore || true
    echo "[restore] done"
    exit 0
fi

TOPK="${1:-6}"
ALPHA_SHARED="${2:-1.2}"
ALPHA_TRANSFER="${3:-0.3}"
TRANSFER_K=3

TAG_RUN="t${TOPK}_s${ALPHA_SHARED//./}_x${ALPHA_TRANSFER//./}"
SERVED_NAME="98e_v4_t18_${TAG_RUN}_nvfp4a16"
RESULTS_DIR="eval_results_vllm_suite/v4_t18/${TAG_RUN}"
LOG_DIR="logs/t18_v4_$(date +%Y%m%d_%H%M%S)_${TAG_RUN}"
SUMMARY="logs/t18_v4_summary.tsv"
TEMPLATES=("gsm8k_30" "humanevalplus_30" "lcb_medium_15")

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

echo "===== T18 Step 1 on v4 — topk=$TOPK α_shared=$ALPHA_SHARED α_transfer=$ALPHA_TRANSFER ====="
echo "  NVFP:      $NVFP"
echo "  drop-map:  $DROP_MAP"
echo "  served-as: $SERVED_NAME"
echo "  results:   $RESULTS_DIR"
echo "  log dir:   $LOG_DIR"
echo

# --- Preflight ---
[ -d "$NVFP" ] || { echo "[FATAL] $NVFP not found"; exit 1; }
[ -f "$DROP_MAP" ] || { echo "[FATAL] $DROP_MAP not found"; exit 1; }
[ -d "$BASE_HF" ] || { echo "[FATAL] $BASE_HF not found"; exit 1; }
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee "$LOG_DIR/gpu_before.txt"

# --- 1) top-k dial ---
echo "[1/4] router_topk_dial.py --top-k $TOPK"
/root/anaconda3/envs/omnimergekit/bin/python scripts/router_topk_dial.py \
    --model-dir "$NVFP" --top-k "$TOPK" 2>&1 | tee "$LOG_DIR/01_topk.log"

# --- 2) shared expert upweight ---
echo "[2/4] router_shared_upweight.py --alpha $ALPHA_SHARED"
/root/anaconda3/envs/omnimergekit/bin/python scripts/router_shared_upweight.py \
    --model-dir "$NVFP" --alpha "$ALPHA_SHARED" 2>&1 | tee "$LOG_DIR/02_shared.log"

# --- 3) soft router transfer ---
echo "[3/4] router_soft_transfer.py --alpha $ALPHA_TRANSFER --top-k $TRANSFER_K"
/root/anaconda3/envs/omnimergekit/bin/python scripts/router_soft_transfer.py \
    --base-dir "$BASE_HF" --variant-dir "$NVFP" --drop-map "$DROP_MAP" \
    --alpha "$ALPHA_TRANSFER" --top-k "$TRANSFER_K" 2>&1 | tee "$LOG_DIR/03_soft.log"

# --- 4) re-smoke C2-comparable triad ---
eval_one() {
    local TPL=$1
    echo "[4/4 $TPL] $(date +%H:%M:%S) start"
    PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH PYTHONDONTWRITEBYTECODE=1 \
    VLLM_PYTHON=/root/anaconda3/envs/vllm/bin/python \
    /root/anaconda3/envs/omnimergekit/bin/python /shared/dev/omnimergekit/eval/omk_eval.py \
        --model "$NVFP" \
        --template "$TPL" \
        --backend vllm \
        --port "$PORT" \
        --served-name "$SERVED_NAME" \
        --tokenizer "$BASE_HF" \
        --max-model-len 40960 \
        --results-dir "$RESULTS_DIR" \
        2>&1 | tee -a "$LOG_DIR/04_${TPL}.log"
    echo "[4/4 $TPL] $(date +%H:%M:%S) done"
}

for TPL in "${TEMPLATES[@]}"; do
    eval_one "$TPL"
done

# --- score parse + summary ---
parse_lm_eval() {
    local TPL=$1 METRIC=$2
    local R_DIR="$RESULTS_DIR/$TPL/$TPL/$SERVED_NAME/lm_eval_out/$SERVED_NAME"
    local R=$(ls -t "$R_DIR"/results_*.json 2>/dev/null | head -1 || true)
    [ -z "$R" ] && { echo "?"; return; }
    /root/anaconda3/envs/omnimergekit/bin/python -c "
import json
r = json.load(open('$R'))['results']
v = list(r.values())[0]
for k in ['$METRIC', '$METRIC,extract_chat', '$METRIC,strict-match', '$METRIC,flexible-extract']:
    if k in v:
        print(v[k]); break
else:
    print('?')
"
}
parse_lcb() {
    local TPL=$1
    local R=$(ls -t "$RESULTS_DIR/$TPL/$TPL/$SERVED_NAME"/lcb_result*.json 2>/dev/null | head -1 || true)
    [ -z "$R" ] && { echo "?"; return; }
    /root/anaconda3/envs/omnimergekit/bin/python -c "
import json
r = json.load(open('$R'))
print(r.get('pass_at_1', r.get('pass@1', '?')))
"
}

GS=$(parse_lm_eval "gsm8k_30" "exact_match")
HE=$(parse_lm_eval "humanevalplus_30" "pass@1")
LC=$(parse_lcb "lcb_medium_15")

[ -f "$SUMMARY" ] || echo -e "ts\ttopk\ta_shared\ta_transfer\tgsm_30\the+30\tlcb15\tnotes" > "$SUMMARY"
echo -e "$(date -Iseconds)\t$TOPK\t$ALPHA_SHARED\t$ALPHA_TRANSFER\t$GS\t$HE\t$LC\t" >> "$SUMMARY"

echo
echo "===== T18 Step 1 on v4 DONE — $TAG_RUN ====="
echo "  v4 anchors:  gsm_30 26/30 (0.867)  HE+/30 25/30 (0.833)  LCB-15 9/15 (0.600)"
echo "  C2 result:   gsm_30 22/30 (0.733)  HE+/30 28/30 (0.933)  LCB-15 13/15 (0.867)"
echo "  this run:    gsm_30=$GS  HE+/30=$HE  LCB-15=$LC"
echo
echo "Summary row appended: $SUMMARY"
echo
echo "Restore before next config:  bash $(basename "$0") --restore"
