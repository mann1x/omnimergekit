#!/usr/bin/env bash
# gpu0_fold_std16.sh — fold GPU0 into the running STD16 cohort gate as a 3rd worker once GPU0
# frees (128e gate + orchestrator done). Uses the SAME atomic claim-lock pool as the two GPU1
# workers (mkdir locks/$T.lock against /mnt/sdc/ml/std16_gate) so it can never double-claim a
# tier. Mirrors gate_std16_cohort.sh worker() verbatim but GPU=0, port=8220. PID-kill only
# (gate_sweep48 traps its own server). Nothing destructive.
set -uo pipefail
GATE=/srv/ml/scripts/gate_sweep48_minp_p_b9700.sh
GG=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF
STEM=gemma-4-A4B-98e-v7-coder-it
WORK=/mnt/sdc/ml/std16_gate
SUMMARY="$WORK/SUMMARY.tsv"
GPU=0; PORT=8220
TIERS=(Q6_K Q2_K_L Q3_K_S Q3_K_M Q3_K_L Q3_K_XL IQ4_XS IQ4_NL Q4_0 Q4_1 \
       Q4_K_S Q4_K_M Q4_K_L Q5_K_S Q5_K_M Q5_K_L Q6_K_L Q8_0)
ts(){ date '+%T %Z'; }
LOG=/mnt/sdc/ml/t223_fk/gpu0_fold_std16.log
exec >>"$LOG" 2>&1
echo "==================== gpu0 fold worker start $(ts) ===================="

# wait for GPU0 to free (128e gate + orchestrator done): mem<3000 MB stable across 2 polls
free=0
for i in $(seq 1 300); do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null | tr -d ' ')
  if [ -n "$m" ] && [ "$m" -lt 3000 ]; then free=$((free+1)); else free=0; fi
  [ "$free" -ge 2 ] && { echo "[$(ts)] GPU$GPU free (mem=${m}MB) — joining STD16 gate pool on :$PORT"; break; }
  sleep 20
done
[ "$free" -ge 2 ] || { echo "[$(ts)] GPU$GPU never freed; abort fold"; exit 1; }

for T in "${TIERS[@]}"; do
  [ -f "$WORK/done/$T.done" ] && continue
  mkdir "$WORK/locks/$T.lock" 2>/dev/null || continue          # atomic claim — same pool as GPU1
  gguf="$GG/${STEM}-${T}.gguf"
  if [ ! -f "$gguf" ]; then echo "[$(ts)] [:$PORT] SKIP $T (no gguf)"; continue; fi
  echo "[$(ts)] [:$PORT GPU0] GATE $T start"
  bash "$GATE" "$gguf" "$GPU" "$PORT" "$WORK/out/${T}.json" "STD16-$T" \
    > "$WORK/logs/${T}.log" 2>&1 || echo "[$(ts)] [:$PORT] $T gate rc=$?"
  fails=$(grep -oE "fails=[0-9]+/[0-9]+" "$WORK/logs/${T}.log" 2>/dev/null | paste -sd' ')
  loopers=$(grep -cE "FAIL=True" "$WORK/logs/${T}.log" 2>/dev/null)
  printf "%s\t%s\tFAILlines=%s\n" "$T" "${fails:-PARSE_FAIL}" "${loopers:-0}" >> "$SUMMARY"
  echo "[$(ts)] [:$PORT GPU0] GATE $T DONE  ${fails:-?}"
  touch "$WORK/done/$T.done"
done
echo "[$(ts)] ==================== gpu0 fold worker DONE ===================="
echo "GPU0_FOLD_STD16_DONE"
