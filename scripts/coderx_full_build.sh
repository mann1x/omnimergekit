#!/usr/bin/env bash
# coderx_full_build.sh — full GGUF tier build (NO UPLOAD) for the v7-coderx re-release.
#
# Reuses the F16 + preserved imatrix already in gguf_coderx/ (built by the loop-cmp run),
# so this is pure-CPU and fast — no GPU, no imatrix recompute. Builds the DEFAULT tier
# set (K/IQ family). CD-* tiers, qat-Q4_0, and NVFP4A16 are SEPARATE follow-on stages
# (CD needs coderx CD maps; qat needs the QAT base; NVFP4A16 needs modelopt+GPU).
#
# HARD GATE: this script NEVER uploads. HF/ollama upload is gated on card sign-off.
set -uo pipefail
BF16=/mnt/sdc/ml/cx_std16/CX16c4l3-bf16
OUTDIR=/mnt/sdc/ml/cx_std16/gguf_coderx
QG=/srv/ml/repos/omnimergekit/scripts/quantize_gguf.py
PYB=/root/anaconda3/envs/omnimergekit/bin/python
LOG=/mnt/sdc/ml/coderx_full_build.log
LOCK=/mnt/sdc/ml/coderx_full_build.lock
ts(){ date '+%T %Z'; }
magic_ok(){ [ -f "$1" ] && [ "$(head -c4 "$1" 2>/dev/null)" = "GGUF" ]; }

exec >>"$LOG" 2>&1
exec 9>"$LOCK"; flock -n 9 || { echo "[$(ts)] already running (lock held) — abort"; exit 0; }
echo "################ coderx FULL tier build (no-upload) $(ts) ################"

# ---- preflight: inputs from the loop-cmp build must be present ----
[ -d "$BF16" ] || { echo "FATAL no bf16 $BF16"; exit 2; }
magic_ok "$OUTDIR/CX16c4l3-bf16-F16.gguf" || { echo "FATAL no reusable F16 in $OUTDIR"; exit 2; }
[ -f "$OUTDIR/imatrix.dat" ] || { echo "FATAL no imatrix.dat in $OUTDIR"; exit 2; }
free=$(df -P /mnt/sdc | awk 'NR==2{print $4}'); [ "$free" -gt 250000000 ] || { echo "FATAL disk<250G ($free K)"; exit 2; }

# ---- default tier set, no-upload, reuse F16+imatrix (GPUs idle now -> 24 threads) ----
THREADS=24
echo "[$(ts)] quantize_gguf --no-upload (DEFAULT tiers, reuse F16+imatrix, CPU threads=$THREADS)"
OMK_NO_README=1 nice -n 10 "$PYB" "$QG" --model "$BF16" --output-dir "$OUTDIR" \
    --base-precision f16 --no-upload \
    --base-model-id ManniX-ITA/gemma-4-A4B-98e-v7-coderx-it --threads "$THREADS"
rc=$?; echo "[$(ts)] quantize_gguf exit=$rc"

echo "=== built GGUF tiers ($(ts)) ==="
for f in "$OUTDIR"/*.gguf; do
  magic_ok "$f" && st=OK || st=BAD
  printf "  %-7s %10s  %s\n" "$st" "$(stat -c %s "$f" 2>/dev/null | numfmt --to=iec)" "$(basename "$f")"
done
echo "###### CODERX_FULLBUILD_DONE $(ts) rc=$rc ######"
