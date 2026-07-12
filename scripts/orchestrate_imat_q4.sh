#!/usr/bin/env bash
# orchestrate_imat_q4.sh — quantize dern11 imat-Q4_K_M (reusing the saved dern11-imatrix.dat)
# and run it through the agentic loop gate across the FULL temp grid {0.8,0.85,0.9,0.95}.
# Tests whether the imatrix that halved the Q6 loop (while keeping thinking on) behaves the
# same on Q4 — i.e. does it recover Q4's suppressed think branch (think>0) and if so does the
# loop return, vs noimat-Q4 which is clean at 0.85-0.9 BUT think=0 (quantization artifact).
# GPU0, sequential. imatrix.dat already preserved next to the quants.
set -uo pipefail
SFT=/mnt/sdc/ml/sft_heal
BIN=/mnt/sdc/ml/llama.cpp-latest/build/bin
DUMP=/mnt/sdc/ml/llama.cpp-latest/gguf-py/gguf/scripts/gguf_dump.py
PY=/root/anaconda3/envs/omnimergekit/bin/python
F16=$SFT/v7coder-dern11-F16.gguf
IMAT=$SFT/dern11-imatrix.dat
Q4=$SFT/gemma-4-A4B-98e-v7-coder-dern11-it-imat-Q4_K_M.gguf
AL=/srv/ml/agentic_loop
GPU=0
ts(){ date '+%T %Z'; }
echo "==================== orchestrate_imat_q4 $(ts) ===================="
for f in "$BIN/llama-quantize" "$F16" "$IMAT"; do [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }; done

# 1) quantize imat-Q4_K_M (reuse existing imatrix)
if [ ! -e "$Q4" ]; then
  echo "[$(ts)] quantize imat-Q4_K_M (--imatrix dern11-imatrix.dat)"
  CUDA_VISIBLE_DEVICES=$GPU "$BIN/llama-quantize" --imatrix "$IMAT" "$F16" "$Q4" Q4_K_M 32 \
    > "$SFT/dern11_imat_q4_quant.log" 2>&1 \
    || { echo "FATAL quant"; tail -25 "$SFT/dern11_imat_q4_quant.log"; exit 4; }
fi
echo "[$(ts)] imat-Q4 metadata (expect file_type 15 + imatrix keys):"
"$PY" "$DUMP" --no-tensors "$Q4" 2>/dev/null | grep -iE "file_type|imatrix.dataset|imatrix.chunks" | head

# 2) loop gate — full temp grid (two passes: {0.9,0.8} then {0.95,0.85})
echo "[$(ts)] loop gate imat-Q4 {t0.9,t0.8} on GPU$GPU:8190"
bash /srv/ml/scripts/gate_sweep48_minp_p.sh "$Q4" 0 8190 \
  "$AL/results/dern11_imatq4_minp48.json" dern11-imatQ4
echo "[$(ts)] loop gate imat-Q4 {t0.95,t0.85} on GPU$GPU:8190"
MATRIX=matrix_minp_2temp_b.json bash /srv/ml/scripts/gate_sweep48_minp_p.sh "$Q4" 0 8190 \
  "$AL/results/dern11_imatq4_minp_t9585.json" dern11-imatQ4-t9585
echo "[$(ts)] === ORCHESTRATE_IMAT_Q4 DONE ==="