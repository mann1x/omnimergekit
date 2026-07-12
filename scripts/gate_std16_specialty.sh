#!/usr/bin/env bash
# gate_std16_specialty.sh — 48-seed deploy-sampler loop gate for the 3 STD16 specialty tiers
# (CD-Q2_K, qat-Q4_0, CD-qat-Q4_K_M). Self-driving: waits for the plain loop gate to finish
# (no GPU contention) AND for all 3 specialty GGUFs to be built, then gates CD-Q2_K + qat-Q4_0
# in parallel (GPU0/GPU1) and CD-qat-Q4_K_M after. Same gate_sweep48 b9700 / vendor_minp_rep
# (t0.9/t0.8) as the plain cohort — apples-to-apples loop numbers.
set -uo pipefail
GATE=/srv/ml/scripts/gate_sweep48_minp_p_b9700.sh
GG=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF
STEM=gemma-4-A4B-98e-v7-coder-it
WORK=/mnt/sdc/ml/std16_gate/specialty
mkdir -p "$WORK/out" "$WORK/logs"
SUMMARY="$WORK/SUMMARY.tsv"
DONE=/mnt/sdc/ml/std16_gate/done
PLAIN_NEED="Q4_K_S Q4_K_M Q4_K_L Q5_K_M Q5_K_L Q6_K_L Q8_0"
TIERS=(CD-Q2_K qat-Q4_0 CD-qat-Q4_K_M)
LOG="$WORK/gate_specialty.log"
exec >>"$LOG" 2>&1
ts(){ date -u +%T; }
echo "==================== specialty gate start $(ts) UTC ===================="

# wait for plain loop gate to finish (frees GPUs)
for i in $(seq 1 900); do
  miss=""; for T in $PLAIN_NEED; do [ -f "$DONE/$T.done" ] || miss="$miss$T "; done
  [ -z "$miss" ] && { echo "[$(ts)] plain gate done"; break; }
  sleep 30
done
# wait for all 3 specialty GGUFs to be built
for i in $(seq 1 900); do
  ok=1; for T in "${TIERS[@]}"; do [ -s "$GG/$STEM-$T.gguf" ] || ok=0; done
  [ "$ok" = 1 ] && { echo "[$(ts)] all 3 specialty GGUFs present"; break; }
  sleep 30
done

gate_one(){ # tier gpu port
  local T="$1" G="$2" P="$3"
  local gguf="$GG/$STEM-$T.gguf"
  [ -f "$WORK/$T.done" ] && { echo "[$(ts)] $T already done"; return 0; }
  [ -s "$gguf" ] || { echo "[$(ts)] SKIP $T (no gguf)"; return 0; }
  echo "[$(ts)] [GPU$G:$P] GATE $T start"
  bash "$GATE" "$gguf" "$G" "$P" "$WORK/out/$T.json" "STD16-$T" > "$WORK/logs/$T.log" 2>&1 || echo "[$(ts)] $T gate rc=$?"
  local arm; arm=$(grep -E "fails=[0-9]+/48.*loops=" "$WORK/logs/$T.log" 2>/dev/null | sed "s/^[[:space:]]*//" | paste -sd" | ")
  printf "%s\t%s\n" "$T" "${arm:-PARSE_FAIL}" >> "$SUMMARY"
  echo "[$(ts)] [GPU$G:$P] GATE $T DONE  ${arm:-?}"
  touch "$WORK/$T.done"
}

# CD-Q2_K + qat-Q4_0 in parallel, then CD-qat-Q4_K_M
gate_one CD-Q2_K 0 8240 &
gate_one qat-Q4_0 1 8241 &
wait
gate_one CD-qat-Q4_K_M 0 8240

echo "[$(ts)] ==================== specialty gate DONE ===================="
echo "SPECIALTY_GATE_DONE"
echo "--- specialty SUMMARY ---"; cat "$SUMMARY" 2>/dev/null
