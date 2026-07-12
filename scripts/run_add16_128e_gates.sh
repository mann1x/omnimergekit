#!/usr/bin/env bash
# run_add16_128e_gates.sh — fires on GPU0 once the 12B test frees it. Sequential pipeline:
#   STEP 1: ADD16-Q3_K_S 48-seed loop gate @ vendor_minp_rep (t0.9/t0.8) — vs STD16 Q3_K_S 19/16
#   STEP 2: build the unpruned 128e imat-Q3_K_S (matched imatrix recipe)
#   STEP 3: 128e-Q3_K_S 48-seed gate @ google/vendor_base (temp 1.0) — quant-loop base reference
# GPU0-only on purpose (GPU1 is owned by the STD16 gate; never collide with it). PID-kill only
# (the gate wrapper traps+kills its own server). Nothing here is destructive.
set -uo pipefail
GPU=0; P1=8230; P2=8231
T=/mnt/sdc/ml/t223_fk
GATE=/srv/ml/scripts/gate_sweep48_minp_p_b9700.sh
B128=/srv/ml/scripts/build_128e_q3ks_baseref.sh
ADD16=$T/ADD16-Q3_K_S.gguf
BREF=$T/128e-imat-Q3_K_S.gguf
LOG=$T/add16_128e_gates.log
ts(){ date '+%T %Z'; }
exec >>"$LOG" 2>&1
echo "==================== orchestrator start $(ts) (target GPU$GPU) ===================="

# --- wait for the ADD16-Q3_K_S build to finish (CPU, ~5 min) ---
for i in $(seq 1 240); do
  [ -f "$ADD16" ] && grep -q ADD16_Q3KS_BUILD_DONE "$T/add16_q3ks_build.log" 2>/dev/null && break
  sleep 15
done
[ -f "$ADD16" ] || { echo "[$(ts)] FATAL ADD16 build never finished"; exit 1; }
echo "[$(ts)] ADD16-Q3_K_S ready: $(stat -c%s "$ADD16") bytes"

# --- wait for GPU0 to free (12B test done): mem.used < 3000 MB, stable across 2 polls ---
free=0
for i in $(seq 1 540); do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null | tr -d ' ')
  if [ -n "$m" ] && [ "$m" -lt 3000 ]; then free=$((free+1)); else free=0; fi
  [ "$free" -ge 2 ] && { echo "[$(ts)] GPU$GPU free (mem=${m}MB)"; break; }
  sleep 20
done
[ "$free" -ge 2 ] || { echo "[$(ts)] FATAL GPU$GPU never freed"; exit 1; }

# --- STEP 1: ADD16-Q3_K_S gate @ vendor_minp_rep ---
echo "[$(ts)] === STEP 1: ADD16-Q3_K_S gate (vendor_minp_rep t0.9/t0.8) ==="
bash "$GATE" "$ADD16" $GPU $P1 results/ADD16-Q3_K_S.json ADD16-Q3_K_S \
  || echo "[$(ts)] WARN ADD16 gate returned nonzero"

# --- STEP 2: build 128e imat-Q3_K_S (matched imatrix) ---
echo "[$(ts)] === STEP 2: build 128e-Q3_K_S base reference (matched imatrix) ==="
if [ ! -f "$BREF" ]; then
  bash "$B128" $GPU || { echo "[$(ts)] FATAL 128e build failed"; exit 1; }
else
  echo "[$(ts)] 128e-Q3_K_S already built, skipping"
fi

# --- STEP 3: 128e-Q3_K_S gate @ google/vendor_base ---
echo "[$(ts)] === STEP 3: 128e-Q3_K_S gate (google/vendor_base temp 1.0) ==="
MATRIX=matrix_gate_vendor_base.json bash "$GATE" "$BREF" $GPU $P2 results/128e-Q3_K_S-vbase.json 128e-Q3_K_S-vbase \
  || echo "[$(ts)] WARN 128e gate returned nonzero"

echo "==================== orchestrator DONE $(ts) ===================="
