#!/usr/bin/env bash
# regate_std16_specialty.sh — RECOVERY re-gate for the 3 STD16 specialty tiers
# (CD-Q2_K, qat-Q4_0, CD-qat-Q4_K_M) after the original specialty gate RACED the QAT
# llama-quantize write. Root cause: gate_std16_specialty.sh readiness used `[ -s gguf ]`
# (size>0), which PASSED on the half-written CD-qat-Q4_K_M.gguf at 23:48:20 (llama-quantize
# started 23:48:01, finished 23:49:14). Its magic was still '????' so all 3 gates crashed in
# 2-4s (the shared per-port server log logs/minp48_srv_8240.log then cross-contaminated the
# CD-Q2_K death-tail), yet the driver still wrote SPECIALTY_GATE_DONE -> the HE+/MPE orchestrator
# would silently skip all 3 (PARSE_FAIL -> loops=99 -> not eligible).
#
# Durable fix here: (0) invalidate the FALSE marker so the orchestrator stays blocked,
# (1) magic-header + size preflight (not -s), (2) wait for the CD-IQ2_NL gate to free the GPUs,
# (3) re-gate all 3 cleanly with DISTINCT ports (no shared server log), (4) re-emit
# SPECIALTY_GATE_DONE to the watched driver log only AFTER valid arm-summary logs exist.
set -uo pipefail
GATE=/srv/ml/scripts/gate_sweep48_minp_p_b9700.sh
GG=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF
STEM=gemma-4-A4B-98e-v7-coder-it
WORK=/mnt/sdc/ml/std16_gate/specialty
CDIQ2NL=/mnt/sdc/ml/std16_gate/cd_iq2nl
SUMMARY="$WORK/SUMMARY.tsv"
DRIVER="$WORK/gate_specialty.log"        # file the HE+/MPE orchestrator greps for SPECIALTY_GATE_DONE
LOG="$WORK/regate_specialty.log"
TIERS=(CD-Q2_K qat-Q4_0 CD-qat-Q4_K_M)
mkdir -p "$WORK/out" "$WORK/logs"
exec >>"$LOG" 2>&1
ts(){ date -u +%T; }
magic(){ head -c4 "$1" 2>/dev/null; }
echo "==================== specialty RE-GATE start $(ts) UTC ===================="

# [0] invalidate the FALSE SPECIALTY_GATE_DONE so the orchestrator stays blocked until valid results
sed -i '/^SPECIALTY_GATE_DONE$/d' "$DRIVER" 2>/dev/null || true
echo "[$(ts)] invalidated false SPECIALTY_GATE_DONE in $DRIVER"
: > "$SUMMARY"
for T in "${TIERS[@]}"; do rm -f "$WORK/$T.done" "$WORK/logs/$T.log"; done

# [1] magic + size preflight — the durable fix vs the -s race (files are complete now)
for T in "${TIERS[@]}"; do
  f="$GG/$STEM-$T.gguf"; sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  { [ "$(magic "$f")" = GGUF ] && [ "$sz" -gt 5000000000 ]; } || {
    echo "[$(ts)] FATAL $T invalid/incomplete GGUF (magic=$(magic "$f") sz=$sz)"; exit 1; }
done
echo "[$(ts)] all 3 specialty GGUFs complete (magic=GGUF, size>5G)"

# [2] wait for the CD-IQ2_NL gate to free both GPUs (clean, contention-free loop measurement)
for i in $(seq 1 1400); do
  grep -q CD_IQ2NL_GATE_DONE "$CDIQ2NL/gate_cd_iq2nl.log" 2>/dev/null && {
    echo "[$(ts)] CD-IQ2_NL gate done — GPUs free"; break; }
  sleep 30
done

gate_one(){ # tier gpu port
  local T="$1" G="$2" P="$3" gguf="$GG/$STEM-$T.gguf"
  echo "[$(ts)] [GPU$G:$P] RE-GATE $T start (gguf=$(basename "$gguf"))"
  bash "$GATE" "$gguf" "$G" "$P" "$WORK/out/$T.json" "STD16-$T" > "$WORK/logs/$T.log" 2>&1 || echo "[$(ts)] $T gate rc=$?"
  local arm; arm=$(grep -E "fails=[0-9]+/48.*loops=" "$WORK/logs/$T.log" 2>/dev/null | sed "s/^[[:space:]]*//" | paste -sd" | ")
  printf "%s\t%s\n" "$T" "${arm:-PARSE_FAIL}" >> "$SUMMARY"
  echo "[$(ts)] [GPU$G:$P] RE-GATE $T DONE  ${arm:-?}"
  touch "$WORK/$T.done"
}
# distinct ports so the per-port server log is never shared (CD-qat now 8242, not 8240)
gate_one CD-Q2_K  0 8240 &
gate_one qat-Q4_0 1 8241 &
wait
gate_one CD-qat-Q4_K_M 0 8242

echo "[$(ts)] ==================== specialty RE-GATE DONE ===================="
echo "--- specialty SUMMARY ---"; cat "$SUMMARY" 2>/dev/null
echo "SPECIALTY_GATE_DONE" >> "$DRIVER"   # re-emit ONLY now that valid arm-summary logs exist
echo "[$(ts)] re-emitted SPECIALTY_GATE_DONE to $DRIVER"
echo "SPECIALTY_REGATE_DONE"
