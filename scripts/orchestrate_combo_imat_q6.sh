#!/usr/bin/env bash
# orchestrate_combo_imat_q6.sh — T206 imat arm: imatrix-Q6 for the eogfk+dern11 combo.
#
# Runs IN PARALLEL on GPU1 with the noimat build/gate (GPU0). Waits for the combo F16 to
# finish converting (signalled by the combo log reaching the quantize step or FATAL), then:
#   1. compute a FRESH combo-specific imatrix (calib_both.txt, 128 chunks) — imatrix is
#      model-specific, dern11-imatrix.dat is NOT reusable. Preserved next to the quant.
#   2. quantize imat-Q6_K (--imatrix)
#   3. loop gate vendor_minp_rep {0.9, 0.8} on GPU1:8191
#
# Compare combo-imat-Q6 vs dern11-imat-Q6 (t0.9=1/48, t0.8=2/48) — the best dern11 arm —
# and combo-Q6_K-noimat (GPU0 arm) to see if the EOG force-keep stacks with imatrix cleaning.
set -uo pipefail
BIN=/mnt/sdc/ml/llama.cpp-latest/build/bin
DUMP=/mnt/sdc/ml/llama.cpp-latest/gguf-py/gguf/scripts/gguf_dump.py
PY=/root/anaconda3/envs/omnimergekit/bin/python
SFT=/mnt/sdc/ml/sft_heal
F16=$SFT/v7coder-eogfk-dern11-F16.gguf
IMAT=$SFT/eogfk-dern11-imatrix.dat
Q6=$SFT/gemma-4-A4B-98e-v7-coder-eogfk-dern11-it-imat-Q6_K.gguf
CALIB=/mnt/sdc/ml/qat_investig/calib_both.txt
COMBO_LOG=$SFT/eogfk_dern11_combo.log
GATE=/srv/ml/scripts/gate_sweep48_minp_p.sh
AL=/srv/ml/agentic_loop
GPU=1
ts(){ date '+%T %Z'; }
echo "==================== orchestrate_combo_imat_q6 $(ts) ===================="

# 0. wait for the combo F16 to be complete (noimat pipeline reached the quant step) or bail on FATAL
echo "[$(ts)] waiting for combo F16 (combo log -> 'quantize Q6_K ->' or FATAL) ..."
while true; do
  if grep -qE "^FATAL|FATAL " "$COMBO_LOG" 2>/dev/null; then echo "[$(ts)] combo build FATAL — aborting imat arm"; exit 1; fi
  if grep -q "quantize Q6_K ->" "$COMBO_LOG" 2>/dev/null && [ -e "$F16" ]; then break; fi
  sleep 60
done
echo "[$(ts)] combo F16 present: $(du -h "$F16" | cut -f1)"

for f in "$BIN/llama-imatrix" "$BIN/llama-quantize" "$F16" "$CALIB" "$GATE"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
export PATH="$(dirname "$PY"):$PATH"

# 1. compute fresh combo imatrix (GPU1) — PRESERVED next to the quants
if [ ! -e "$IMAT" ]; then
  echo "[$(ts)] imatrix: -ngl 99 --chunks 128 -f calib_both.txt (GPU$GPU)"
  CUDA_VISIBLE_DEVICES=$GPU "$BIN/llama-imatrix" -m "$F16" -f "$CALIB" -o "$IMAT" -ngl 99 --chunks 128 \
    > "$SFT/eogfk_dern11_imatrix_build.log" 2>&1 \
    || { echo "FATAL imatrix"; tail -25 "$SFT/eogfk_dern11_imatrix_build.log"; exit 3; }
fi
echo "[$(ts)] imatrix.dat $(stat -c%s "$IMAT" 2>/dev/null) bytes"

# 2. quantize imat-Q6_K
if [ ! -e "$Q6" ]; then
  echo "[$(ts)] quantize imat-Q6_K (--imatrix)"
  CUDA_VISIBLE_DEVICES=$GPU "$BIN/llama-quantize" --imatrix "$IMAT" "$F16" "$Q6" Q6_K 32 \
    > "$SFT/eogfk_dern11_imat_q6_quant.log" 2>&1 \
    || { echo "FATAL quant"; tail -25 "$SFT/eogfk_dern11_imat_q6_quant.log"; exit 4; }
fi
echo "[$(ts)] imat-Q6 metadata (expect file_type 18 + imatrix keys):"
"$PY" "$DUMP" --no-tensors "$Q6" 2>/dev/null | grep -iE "file_type|imatrix.dataset|imatrix.chunks" | head

# 3. loop gate {0.9, 0.8} on GPU1
echo "[$(ts)] loop gate combo imat-Q6 {t0.9,t0.8} on GPU$GPU:8191"
bash "$GATE" "$Q6" "$GPU" 8191 "$AL/results/eogfk_dern11_imatq6_minp48.json" eogfk-dern11-imatQ6

echo "[$(ts)] === COMBO IMAT-Q6 DONE ==="
