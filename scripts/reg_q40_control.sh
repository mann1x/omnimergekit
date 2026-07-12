#!/usr/bin/env bash
# reg_q40_control.sh — SINGLE-VARIABLE CONTROL for the qat-Q4_0 loop failure.
#
# qat-Q4_0 (DERN on Google's QAT q4_0-unquantized base, no-imatrix Q4_0) failed the
# 48-seed agentic loop gate at 15/48 (t0.9) and 20/48 (t0.8), and regressed code to
# HE+ 81.10 / MPE 67.67 (vs regular imat-Q6 93.29 / 89.33, 0/48). Two variables changed
# between the clean anchor and the qat result: (Q6 imatrix -> Q4_0 no-imatrix) AND
# (regular-bf16 DERN base -> QAT DERN base). This control holds the base = REGULAR and
# quantizes the SAME regular DERN'd F16 with the IDENTICAL command qat used
# (llama-quantize ... Q4_0 32, no imatrix), then runs the IDENTICAL gate script.
#
# Decomposition (3 points, regular base unless noted):
#   regular imat-Q6     = 0/48   (known anchor)
#   regular Q4_0 no-imat = THIS  (isolates Q6-imatrix vs Q4_0-noimatrix on regular base)
#   qat Q4_0 no-imat     = 15,20 (known; vs THIS isolates regular base vs QAT base)
#
# Reads:  if THIS ~0-2/48  -> Q4_0-noimat is fine on regular base => qat cooking is the
#                              culprit (DERN rewrite kills QAT calibration). Drop qat,
#                              regular tiers vindicated.
#         if THIS ~15-20/48 -> Q4_0/no-imatrix itself loops => anti-loop is imatrix/Q6-
#                              dependent, NOT QAT-specific. Publish-critical: scope the
#                              0/48 claim to imat tiers; gate the :latest (Q4_K_M) tier.
# PID-kill only (the gate script handles its own server lifecycle).
set -uo pipefail
F16=/mnt/sdc/ml/sft_heal/fkbroad-soft2-F16.gguf
OUTD=/mnt/sdc/ml/v8_qat/reg_ctrl
OUT=$OUTD/gemma-4-A4B-98e-v7-coder-reg-Q4_0.gguf
RES=/srv/ml/agentic_loop/results/reg-soft2-Q40_minp48.json
GATE=/srv/ml/scripts/gate_sweep48_minp_p.sh
QBIN=/mnt/sdc/ml/llama.cpp-latest/build/bin/llama-quantize
mkdir -p "$OUTD"
ts(){ date '+%T %Z'; }
echo "==== REG-Q4_0 CONTROL (regular DERN base, Q4_0 no-imat 32t) $(ts) ===="
for f in "$F16" "$GATE" "$QBIN"; do [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }; done
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc '0-9')
echo "[disk $(ts)] ${free}G free on /mnt/sdc"
[ "${free:-0}" -lt 15 ] && { echo "FATAL <15G free, refuse to build"; exit 9; }
if [ ! -f "$OUT" ]; then
  echo "[build $(ts)] llama-quantize Q4_0 32 (NO imatrix) from regular DERN F16"
  "$QBIN" "$F16" "$OUT" Q4_0 32 || { echo "FATAL quant rc=$?"; exit 3; }
fi
echo "[build $(ts)] done: $(du -h "$OUT" | cut -f1)"
echo "[gate $(ts)] 48-seed loop gate v8-reg-Q4_0 GPU1:8294 (control vs qat 15/20)"
bash "$GATE" "$OUT" 1 8294 "$RES" v8-reg-Q4_0 || echo "[gate $(ts)] WARN rc=$?"
echo "###### REG_Q40_CTRL_DONE $(ts) ######"
