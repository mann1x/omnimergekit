#!/bin/bash
# pick_t18_variants.sh — read the v5fixed sweep summary TSV, sort by
# gsm_flex (then gsm_strict, then tag), pick the top-N, and emit ready
# apply_t18_step1.sh invocations for them.
#
# Convention: gsm8k_30 is the math-preservation selector for the sweep
# (HE/LCB smokes are gated by rumination, NOT by aggregator strategy,
# so they're all 0 — they're not usable selectors here. T18 router
# recovery is what then re-tests the chosen variants on code benches).
#
# Usage:
#   bash scripts/pick_t18_variants.sh                       # top-3 default
#   bash scripts/pick_t18_variants.sh 5                     # top-5
#   bash scripts/pick_t18_variants.sh 3 /path/to/_sweep_summary.tsv
#
# Output: prints the top-N rows in flex-desc order, then prints
# `bash scripts/apply_t18_step1.sh <tag>` lines (default knobs:
# top-k=6, alpha_shared=1.2, alpha_transfer=0.3 — change at will).

set -euo pipefail
cd /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models

N="${1:-3}"
TSV="${2:-$(ls -dt logs/v5fixed_sweep_2026* 2>/dev/null | grep -v outer | head -1)/_sweep_summary.tsv}"

if [ ! -f "$TSV" ]; then
    echo "FAIL: TSV not found at $TSV"
    echo "Pass an explicit path as arg 2 if your sweep ran in a different log dir."
    exit 1
fi

echo "=== sweep summary: $TSV ==="
column -ts$'\t' "$TSV"
echo
echo "=== top-$N by gsm_flex (then gsm_strict, then tag) ==="

# Skip header, sort numeric desc on column 8 (gsm_flex) then 7 (gsm_strict)
# then alpha asc on column 1 (tag). Take first N.
PICKS=$(tail -n +2 "$TSV" \
    | sort -t$'\t' -k8,8nr -k7,7nr -k1,1 \
    | head -n "$N" \
    | awk -F'\t' '{printf "%-30s gsm_flex=%s gsm_strict=%s\n", $1, $8, $7}')
echo "$PICKS"
echo

echo "=== ready apply_t18_step1.sh invocations (defaults: topk=6 αshared=1.2 αtransfer=0.3) ==="
tail -n +2 "$TSV" \
    | sort -t$'\t' -k8,8nr -k7,7nr -k1,1 \
    | head -n "$N" \
    | awk -F'\t' '{printf "bash scripts/apply_t18_step1.sh %s\n", $1}'

echo
echo "Notes:"
echo "  * apply_t18_step1.sh runs router_topk_dial + shared_upweight + soft_transfer"
echo "    then re-smokes gsm8k_30 + humaneval_1_smoke + lcb_medium_1_smoke."
echo "  * Each transform is REVERSIBLE via the script's --restore flag — A/B different"
echo "    knob settings on the same NVFP4A16 dir."
echo "  * After Step 1 looks good on a variant, run Step 2 via router_eac_calibrate.py"
echo "    (A/B: WikiText-2 vs logs/diff_corpus.txt)."
