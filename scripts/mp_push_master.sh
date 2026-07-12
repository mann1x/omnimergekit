#!/usr/bin/env bash
# mp_push_master.sh — two parallel arms chasing 0/48 at Q6 (multi-pass dropped as a no-op):
#   GPU0: imat-Q6 of fx-2.0 (best noimat 3/48) — the real 0/48 shot via the proven imatrix lever
#   GPU1: soft top-2 assignment variant on bare dern11 — the requested assignment-axis falsification
# Both loop-gated vendor_minp_rep {0.9,0.8}. Baselines: dern11-imat-Q6 = 1/48,2/48 ; dern11 noimat = 4/48,2/48.
set -uo pipefail
SFT=/mnt/sdc/ml/sft_heal
S=/srv/ml/scripts
ts(){ date '+%T %Z'; }
echo "==================== mp_push (fx2-imat + soft2) $(ts) ===================="

bash "$S/imat_promote.sh" "$SFT/dern11-knob-fx-2.0-it" dern11-knob-fx-2.0 0 8190 > "$SFT/mp_push_fx2imat.log" 2>&1 &
A=$!
bash "$S/dern_soft_variant.sh" soft2 2 1 8191 > "$SFT/mp_push_soft2.log" 2>&1 &
B=$!

wait "$A"; rA=$?
wait "$B"; rB=$?
echo "[$(ts)] arms done: fx2-imat rc=$rA | soft2 rc=$rB"
echo "==================== MP PUSH DONE $(ts) ===================="
echo "----- fx2-imat summary -----"; grep -E "arm summary|fails=" "$SFT/mp_push_fx2imat.log" 2>/dev/null | tail -4 || echo "(none)"
echo "----- soft2 summary -----";    grep -E "arm summary|fails=" "$SFT/mp_push_soft2.log"   2>/dev/null | tail -4 || echo "(none)"
