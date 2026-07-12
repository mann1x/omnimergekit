#!/usr/bin/env bash
# imat_q4km_gate.sh — PUBLISH-CRITICAL loop gate for the shipping :latest tier.
#
# Finding so far: v8's 0/48 anti-loop is an IMAT-Q6 property — a no-imatrix Q4_0 loops
# on the SAME regular DERN weights (matched control). The tier most users pull is
# :latest = imat-Q4_K_M (the tier sweep builds ALL K-tiers WITH the model-specific
# imatrix fkbroad-soft2-imatrix.dat; HE+ 93.90 / MPE 89.67). This gate answers the only
# open question for the "anti-loop" headline: does the imatrix rescue 0/48 at Q4_K_M the
# way it does at Q6, or is 0/48 Q6-only?
#   PASS 0/48  -> :latest is greedy-anti-loop; headline holds for the imatrix tiers.
#   FAIL       -> headline scoped to imat-Q6; lower tiers keep the baked anti-loop sampler.
#
# Builds imat-Q4_K_M from the regular DERN F16 with the SAME imatrix the sweep used,
# then runs the IDENTICAL 48-seed gate. GPU0. PID-kill only (gate handles its server).
set -uo pipefail
F16=/mnt/sdc/ml/sft_heal/fkbroad-soft2-F16.gguf
IMAT=/mnt/sdc/ml/sft_heal/fkbroad-soft2-imatrix.dat
OUTD=/mnt/sdc/ml/v8_qat/reg_ctrl
OUT=$OUTD/gemma-4-A4B-98e-v7-coder-imat-Q4_K_M.gguf
RES=/srv/ml/agentic_loop/results/reg-soft2-imatQ4KM_minp48.json
GATE=/srv/ml/scripts/gate_sweep48_minp_p.sh
QBIN=/mnt/sdc/ml/llama.cpp-latest/build/bin/llama-quantize
mkdir -p "$OUTD"
ts(){ date '+%T %Z'; }
echo "==== imat-Q4_K_M :latest GATE $(ts) ===="
for f in "$F16" "$IMAT" "$GATE" "$QBIN"; do [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }; done
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc '0-9')
echo "[disk $(ts)] ${free}G free"; [ "${free:-0}" -lt 15 ] && { echo "FATAL <15G free"; exit 9; }
if [ ! -f "$OUT" ]; then
  echo "[build $(ts)] llama-quantize Q4_K_M 32 WITH imatrix (matches shipping :latest)"
  "$QBIN" --imatrix "$IMAT" "$F16" "$OUT" Q4_K_M 32 || { echo "FATAL quant rc=$?"; exit 3; }
fi
echo "[build $(ts)] done: $(du -h "$OUT" | cut -f1)"
echo "[gate $(ts)] 48-seed loop gate v8-imat-Q4_K_M GPU0:8296 (publish-critical)"
bash "$GATE" "$OUT" 0 8296 "$RES" v8-imat-Q4_K_M || echo "[gate $(ts)] WARN rc=$?"
echo "###### IMAT_Q4KM_GATE_DONE $(ts) ######"
