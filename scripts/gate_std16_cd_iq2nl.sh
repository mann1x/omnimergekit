#!/usr/bin/env bash
# gate_std16_cd_iq2nl.sh — 48-seed deploy-sampler loop gate for the 2 CD-IQ2_NL variants
# (IQ2_S tail + Q2_K tail). Self-driving: waits for the specialty gate to finish (no GPU
# contention) AND both GGUFs to be built, then gates both in parallel (GPU0/GPU1). Same
# gate_sweep48 b9700 / vendor_minp_rep (t0.9/t0.8) as the rest of the cohort. Writes per-tier
# logs the HE+/MPE orchestrator reads + a CD_IQ2NL_GATE_DONE marker.
set -uo pipefail
GATE=/srv/ml/scripts/gate_sweep48_minp_p_b9700.sh
GG=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF
STEM=gemma-4-A4B-98e-v7-coder-it
SPEC=/mnt/sdc/ml/std16_gate/specialty
WORK=/mnt/sdc/ml/std16_gate/cd_iq2nl
mkdir -p "$WORK/out" "$WORK/logs"
LOG="$WORK/gate_cd_iq2nl.log"
TIERS=(CD-IQ2_NL CD-IQ2_NL-q2k)
exec >>"$LOG" 2>&1
ts(){ date -u +%T; }
echo "==================== CD-IQ2_NL gate start $(ts) UTC ===================="

# wait for specialty gate to finish (frees GPUs)
for i in $(seq 1 1400); do
  grep -q SPECIALTY_GATE_DONE "$SPEC/gate_specialty.log" 2>/dev/null && { echo "[$(ts)] specialty gate done"; break; }
  sleep 30
done
# wait for both CD-IQ2_NL GGUFs to be built
for i in $(seq 1 1400); do
  ok=1; for T in "${TIERS[@]}"; do [ -s "$GG/$STEM-$T.gguf" ] || ok=0; done
  [ "$ok" = 1 ] && { echo "[$(ts)] both CD-IQ2_NL GGUFs present"; break; }
  sleep 30
done

gate_one(){ # tier gpu port
  local T="$1" G="$2" P="$3"
  local gguf="$GG/$STEM-$T.gguf"
  [ -f "$WORK/$T.done" ] && { echo "[$(ts)] $T already done"; return 0; }
  [ -s "$gguf" ] || { echo "[$(ts)] SKIP $T (no gguf)"; return 0; }
  echo "[$(ts)] [GPU$G:$P] GATE $T start"
  bash "$GATE" "$gguf" "$G" "$P" "$WORK/out/$T.json" "STD16-$T" > "$WORK/logs/$T.log" 2>&1 || echo "[$(ts)] $T gate rc=$?"
  echo "[$(ts)] [GPU$G:$P] GATE $T DONE  $(grep -E 'fails=[0-9]+/48.*loops=' "$WORK/logs/$T.log" 2>/dev/null | sed 's/^[[:space:]]*//' | paste -sd' | ')"
  touch "$WORK/$T.done"
}
gate_one CD-IQ2_NL     0 8250 &
gate_one CD-IQ2_NL-q2k 1 8251 &
wait

echo "[$(ts)] ==================== CD-IQ2_NL gate DONE ===================="
echo "CD_IQ2NL_GATE_DONE"
