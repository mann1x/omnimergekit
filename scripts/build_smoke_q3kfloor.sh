#!/usr/bin/env bash
# build_smoke_q3kfloor.sh — test the REAL fix. Plain Q4_K_M (all >=Q4_K) is
# healthy (5/5 STOP); CD with IQ3_S on the FFN/MoE low tier ruminates (5/5). So
# IQ3_S (3.4-bit i-quant) on the pruned 98e experts is the killer, NOT attention.
# Test: swap the low-tier FFN IQ3_S -> Q3_K (robust k-quant, ~3.9-bit), keep
# attention Q5_K, rebuild, termination-smoke. STOP => the fix is IQ3_S->Q3_K.
set -uo pipefail
QUANT=/opt/llama.cpp/build/bin/llama-quantize
GD=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF
F16="$GD/gemma-4-A4B-98e-v7-coder-it-F16.gguf"
IMAT="$GD/imatrix.dat"
SRCMAP=/srv/ml/scripts/cd_maps_fixed_v7coder/tensor_types_CD-Q4_K_M.txt
Q3MAP=/srv/ml/scripts/cd_maps_fixed_v7coder/tensor_types_CD-Q4_K_M_q3kfloor.txt
OUT=/mnt/sdc/ml/quant_sweep_gguf/gemma-4-A4B-98e-v7-coder-it-CD-Q4_K_M-Q3KFLOOR.gguf
LOG=/srv/ml/logs/build_smoke_q3kfloor.txt
: > "$LOG"
for f in "$F16" "$IMAT" "$SRCMAP"; do [ -f "$f" ] || { echo "[FATAL] missing $f" | tee -a "$LOG"; exit 1; }; done

# IQ3_S appears only on the 22 low-tier ffn_* tensors; swap to Q3_K.
sed 's/=IQ3_S/=Q3_K/' "$SRCMAP" > "$Q3MAP"
echo "== map swap IQ3_S->Q3_K ($(grep -c '=Q3_K' "$Q3MAP") Q3_K lines) ==" | tee -a "$LOG"

echo "==== [1/2] rebuild CD-Q4_K_M-Q3KFLOOR from F16 (CPU) $(date -u) ====" | tee -a "$LOG"
"$QUANT" --imatrix "$IMAT" --tensor-type-file "$Q3MAP" "$F16" "$OUT" Q4_K_M >>"$LOG" 2>&1
rc=$?
[ $rc -eq 0 ] && [ -f "$OUT" ] || { echo "[FATAL] quantize rc=$rc" | tee -a "$LOG"; tail -5 "$LOG"; exit 1; }
echo "  built $(du -h "$OUT" | cut -f1) -> $OUT" | tee -a "$LOG"

echo "==== [2/2] termination smoke on GPU0 $(date -u) ====" | tee -a "$LOG"
bash /srv/ml/scripts/smoke_gguf.sh "$OUT" 8263 0 2048 2>&1 | tee -a "$LOG"
echo "[done] $(date -u)" | tee -a "$LOG"
