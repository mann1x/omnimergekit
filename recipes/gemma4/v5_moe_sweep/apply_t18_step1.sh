#!/bin/bash
# apply_t18_step1.sh — apply T18 Step 1 (free knobs) to a v5fixed sweep
# variant and re-smoke.
#
# Pipeline (all idempotent w/ --restore for each script):
#   1. router_topk_dial.py        — set top-k to N
#   2. router_shared_upweight.py  — scale shared mlp by α_shared
#   3. router_soft_transfer.py    — soft-redistribute dropped router rows
#   4. omk_eval                    — humaneval_1_smoke + lcb_medium_1_smoke
#                                    (gsm8k_30 already done in sweep, skip
#                                    unless --resmoke-all)
#
# Each Step 1 transform is REVERSIBLE via the script's --restore flag,
# so multiple α settings can be A/B'd on the same NVFP4A16 dir.
#
# Usage:
#   bash scripts/apply_t18_step1.sh <variant_tag> [<topk>] [<alpha_shared>] [<alpha_transfer>]
#
#   bash scripts/apply_t18_step1.sh A2_lp4_uni 6 1.2 0.3
#       → top-k=6, shared upweight 1.2×, soft transfer α=0.3 top-3 neighbors
#
# Defaults: topk=6 alpha_shared=1.2 alpha_transfer=0.3
#
# Results land in: eval_results_vllm_suite/v5fixed_t18/<variant_tag>__t<topk>_s<aS>_x<aT>/
# Summary appended to:  logs/t18_step1_summary.tsv

set -euo pipefail
cd /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models

TAG="${1:?usage: apply_t18_step1.sh <variant_tag> [topk] [alpha_shared] [alpha_transfer]}"
TOPK="${2:-6}"
ALPHA_SHARED="${3:-1.2}"
ALPHA_TRANSFER="${4:-0.3}"
TRANSFER_K=3

BASE_HF="google/gemma-4-26B-A4B-it"
NVFP="google/gemma-4-A4B-98e-v5fixed-sweep-${TAG}-NVFP4A16"
DROP_MAP="scripts/v5fixed_sweep_${TAG}_drop_map.json"
SERVED_NAME="98e_v5fixed_t18_${TAG}_t${TOPK}_s${ALPHA_SHARED//./}_x${ALPHA_TRANSFER//./}_nvfp4a16"
RESULTS_DIR="eval_results_vllm_suite/v5fixed_t18/${TAG}__t${TOPK}_s${ALPHA_SHARED}_x${ALPHA_TRANSFER}"
LOG_DIR="logs/t18_step1_$(date +%Y%m%d_%H%M%S)_${TAG}"
SUMMARY="logs/t18_step1_summary.tsv"
PORT=8194

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

echo "===== T18 Step 1 — variant=$TAG  topk=$TOPK  α_shared=$ALPHA_SHARED  α_transfer=$ALPHA_TRANSFER ====="
echo "  NVFP:        $NVFP"
echo "  drop-map:    $DROP_MAP"
echo "  served-as:   $SERVED_NAME"
echo "  results dir: $RESULTS_DIR"
echo "  log dir:     $LOG_DIR"
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

# --- 4) re-smoke (HE-1 + LCB-1; gsm8k_30 already in sweep but we re-run because
#       router edits may shift math behavior too) ---
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
}

for TPL in gsm8k_30 humaneval_1_smoke lcb_medium_1_smoke; do
    eval_one "$TPL"
done

# --- score parse + summary ---
parse_score() {
    local TPL=$1
    local R_DIR="$RESULTS_DIR/$TPL/$TPL/$SERVED_NAME/lm_eval_out/$SERVED_NAME"
    local R=$(ls -t "$R_DIR"/results_*.json 2>/dev/null | head -1 || true)
    [ -z "$R" ] && { echo "0 0"; return; }
    /root/anaconda3/envs/omnimergekit/bin/python -c "
import json
r = json.load(open('$R'))['results']
v = list(r.values())[0]
s = v.get('exact_match,strict-match', v.get('pass@1,extract_chat', v.get('pass@1', 0.0)))
f = v.get('exact_match,flexible-extract', s)
print(f'{s} {f}')
"
}

read GSM_S GSM_F < <(parse_score gsm8k_30)
read HE_S HE_F < <(parse_score humaneval_1_smoke)
read LCB_S LCB_F < <(parse_score lcb_medium_1_smoke)

[ -f "$SUMMARY" ] || echo -e "ts\ttag\ttopk\ta_shared\ta_transfer\tgsm_strict\tgsm_flex\the_p1\tlcb_p1\tnotes" > "$SUMMARY"
echo -e "$(date -Iseconds)\t$TAG\t$TOPK\t$ALPHA_SHARED\t$ALPHA_TRANSFER\t$GSM_S\t$GSM_F\t$HE_S\t$LCB_S\t" >> "$SUMMARY"

echo
echo "===== T18 Step 1 DONE — $TAG ====="
echo "  gsm8k_30:           strict=$GSM_S flex=$GSM_F"
echo "  humaneval_1_smoke:  p1=$HE_S"
echo "  lcb_medium_1_smoke: p1=$LCB_S"
echo
echo "Summary row appended: $SUMMARY"
echo
echo "Restore (undo router edits):"
echo "  python scripts/router_topk_dial.py        --model-dir $NVFP --restore"
echo "  python scripts/router_shared_upweight.py  --model-dir $NVFP --restore"
echo "  python scripts/router_soft_transfer.py    --base-dir $BASE_HF --variant-dir $NVFP --drop-map $DROP_MAP --restore"
