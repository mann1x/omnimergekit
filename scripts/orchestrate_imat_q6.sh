#!/usr/bin/env bash
# orchestrate_imat_q6.sh — build a FRESH-imatrix Q6_K of dern11 and run it through the
# exact 48-seed agentic loop gate (vendor_minp_rep x {0.9,0.8}), to test whether an
# imatrix changes the Q6 loop vs the noimat-Q6 baseline (noimat: t0.9=4/48, t0.8=2/48).
# This is a deliberate OFF-POLICY loop probe — Q6_K ships noimat for capability, but the
# imatrix could plausibly shift the think-branch behaviour either way. GPU0, sequential.
# imatrix.dat is SAVED next to the quant (mandatory archival rule).
set -uo pipefail
SFT=/mnt/sdc/ml/sft_heal
BIN=/mnt/sdc/ml/llama.cpp-latest/build/bin
DUMP=/mnt/sdc/ml/llama.cpp-latest/gguf-py/gguf/scripts/gguf_dump.py
PY=/root/anaconda3/envs/omnimergekit/bin/python
F16=$SFT/v7coder-dern11-F16.gguf
CALIB=/mnt/sdc/ml/qat_investig/calib_both.txt
IMAT=$SFT/dern11-imatrix.dat
Q6=$SFT/gemma-4-A4B-98e-v7-coder-dern11-it-imat-Q6_K.gguf
AL=/srv/ml/agentic_loop
GPU=0
ts(){ date '+%T %Z'; }
echo "==================== orchestrate_imat_q6 $(ts) ===================="
for f in "$BIN/llama-imatrix" "$BIN/llama-quantize" "$F16" "$CALIB"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done

# 1) compute imatrix on GPU0 (canonical: -ngl 99, --chunks 128, calib_both.txt)
if [ ! -e "$IMAT" ]; then
  echo "[$(ts)] imatrix: -ngl 99 --chunks 128 -f calib_both.txt  (GPU$GPU)"
  CUDA_VISIBLE_DEVICES=$GPU "$BIN/llama-imatrix" -m "$F16" -f "$CALIB" -o "$IMAT" \
    -ngl 99 --chunks 128 > "$SFT/dern11_imatrix_build.log" 2>&1 \
    || { echo "FATAL imatrix build"; tail -25 "$SFT/dern11_imatrix_build.log"; exit 3; }
fi
[ -s "$IMAT" ] || { echo "FATAL imatrix.dat empty/missing"; exit 3; }
echo "[$(ts)] imatrix.dat $(stat -c %s "$IMAT") bytes"

# 2) quantize imat-Q6_K
if [ ! -e "$Q6" ]; then
  echo "[$(ts)] quantize imat-Q6_K (--imatrix)"
  CUDA_VISIBLE_DEVICES=$GPU "$BIN/llama-quantize" --imatrix "$IMAT" "$F16" "$Q6" Q6_K 32 \
    > "$SFT/dern11_imat_q6_quant.log" 2>&1 \
    || { echo "FATAL quant"; tail -25 "$SFT/dern11_imat_q6_quant.log"; exit 4; }
fi
echo "[$(ts)] imat-Q6 metadata (expect file_type 18 + imatrix keys):"
"$PY" "$DUMP" --no-tensors "$Q6" 2>/dev/null | grep -iE "file_type|imatrix.dataset|imatrix.chunks|imatrix.entries" | head

# 3) 48-seed agentic loop gate, default matrix (vendor_minp_rep x {0.9,0.8})
echo "[$(ts)] loop gate imat-Q6 {t0.9,t0.8} on GPU$GPU:8190"
bash /srv/ml/scripts/gate_sweep48_minp_p.sh "$Q6" 0 8190 \
  "$AL/results/dern11_imatq6_minp48.json" dern11-imatQ6
echo "[$(ts)] === ORCHESTRATE_IMAT_Q6 DONE ==="