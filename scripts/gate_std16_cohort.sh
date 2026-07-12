#!/usr/bin/env bash
# gate_std16_cohort.sh — 48-seed loop gate (vendor_minp_rep t{0.9,0.8}, b9700) on every staged
# STD16 quant tier. Validates the imatrix-everywhere decision per tier: any tier with fails>0 is
# a LOOPER -> flagged for a targeted balanced-imatrix rebuild. 2 concurrent workers on GPU1
# (26B-A4B ~17GB Q6 + 131k ctx fits 2x on 97GB). Resumable: atomic mkdir claim-lock + .done.
# Risky (low-bit) tiers first so decision-relevant results land early. Q6_K = 0/48 anchor control.
set -uo pipefail
GATE=/srv/ml/scripts/gate_sweep48_minp_p_b9700.sh
GG=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF
STEM=gemma-4-A4B-98e-v7-coder-it
WORK=/mnt/sdc/ml/std16_gate; mkdir -p "$WORK/locks" "$WORK/done" "$WORK/out" "$WORK/logs"
SUMMARY="$WORK/SUMMARY.tsv"
GPU=1
ts(){ date '+%T %Z'; }
# Q6_K first = 0/48 anchor control (validates the b9700 setup); then risky low-bit -> safe.
TIERS=(Q6_K Q2_K_L Q3_K_S Q3_K_M Q3_K_L Q3_K_XL IQ4_XS IQ4_NL Q4_0 Q4_1 \
       Q4_K_S Q4_K_M Q4_K_L Q5_K_S Q5_K_M Q5_K_L Q6_K_L Q8_0)

[ -x "$GATE" ] || [ -f "$GATE" ] || { echo "FATAL no gate script $GATE"; exit 2; }

worker() { # $1 = port
  local port="$1" T gguf fails loopers
  for T in "${TIERS[@]}"; do
    [ -f "$WORK/done/$T.done" ] && continue
    mkdir "$WORK/locks/$T.lock" 2>/dev/null || continue        # atomic claim
    gguf="$GG/${STEM}-${T}.gguf"
    if [ ! -f "$gguf" ]; then echo "[$(ts)] [:$port] SKIP $T (no gguf)"; continue; fi
    echo "[$(ts)] [:$port] GATE $T start"
    bash "$GATE" "$gguf" "$GPU" "$port" "$WORK/out/${T}.json" "STD16-$T" \
      > "$WORK/logs/${T}.log" 2>&1 || echo "[$(ts)] [:$port] $T gate rc=$?"
    fails=$(grep -oE "fails=[0-9]+/[0-9]+" "$WORK/logs/${T}.log" 2>/dev/null | paste -sd' ')
    loopers=$(grep -cE "FAIL=True" "$WORK/logs/${T}.log" 2>/dev/null)
    printf "%s\t%s\tFAILlines=%s\n" "$T" "${fails:-PARSE_FAIL}" "${loopers:-0}" >> "$SUMMARY"
    echo "[$(ts)] [:$port] GATE $T DONE  ${fails:-?}  FAIL=True-lines=${loopers:-0}"
    touch "$WORK/done/$T.done"
  done
}

echo "[$(ts)] STD16 LOOP-GATE START — ${#TIERS[@]} tiers, 2 workers GPU$GPU (risky-first)"
worker 8210 &
worker 8211 &
wait
echo "[$(ts)] ===== STD16 LOOP-GATE COMPLETE ====="
echo "=== SUMMARY (tier  fails/48 per arm  FAIL=True-lines) ==="; sort "$SUMMARY" 2>/dev/null
echo "=== LOOPERS (fails>0 -> balanced-imatrix rebuild) ==="
grep -vE "fails=0/48[[:space:]]+fails=0/48" "$SUMMARY" 2>/dev/null | grep -vE "^$" || echo "  (none — all tiers 0/48 clean)"
echo "STD16_GATE_ALL_DONE"
