#!/bin/bash
# Re-run Phase B Router-KD on bs2 with --no-canary to force-save A2_RKD
# Author: claude opus 4.7  2026-05-29
set -uo pipefail
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=$BM/logs/router_kd_force_save_$TS
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/chain.log
exec > >(tee "$LOG") 2>&1

TEACHER=$BM/google/gemma-4-26B-A4B-it
A2_EAC=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-it
A2_RKD=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-it
KD_CORPUS=$BM/scripts/router_calib_corpus.jsonl

echo "[$(date -Iseconds)] === Force-save A2_RKD (--no-canary) ==="
echo "  teacher  : $TEACHER"
echo "  student  : $A2_EAC (input)"
echo "  rkd out  : $A2_RKD"
echo "  KD LR/steps : 1e-5 / 100 (same recipe as gate-failed run)"
echo

# A2_EAC must exist and not be hardlinked
if [ ! -d "$A2_EAC" ]; then
    echo "FATAL: $A2_EAC missing"; exit 2
fi
nlink=$(stat -c '%h' "$A2_EAC/model-00001-of-00006.safetensors")
echo "  A2_EAC shard nlink=$nlink (must be 1 for isolation)"
[ "$nlink" -ne 1 ] && { echo "FATAL: hardlinked input"; exit 2; }

$PY $BM/scripts/router_kd.py \
    --base-dir "$TEACHER" \
    --variant-dir "$A2_EAC" \
    --out-dir "$A2_RKD" \
    --teacher-load bf16 --student-load bf16 \
    --teacher-device '{"":0}' --student-device '{"":1}' \
    --tau 1.0 --lr 1e-5 --max-steps 100 \
    --batch-size 2 --grad-accum 4 \
    --max-seq-len 512 --max-samples 800 \
    --corpus-file "$KD_CORPUS" \
    --checkpoint-dir $LOG_DIR/ckpt \
    --canary-file /srv/ml/scripts/ifeval_rumination_canaries.json \
    --no-canary 2>&1
RC_KD=$?

echo
echo "[$(date -Iseconds)] === Done ==="
echo "  KD exit: $RC_KD"
if [ -d "$A2_RKD" ]; then
    echo "  Wrote: $A2_RKD"
    ls "$A2_RKD" | head -10
    echo "  size:"
    du -sh "$A2_RKD"
    echo "  main shard nlink:"
    stat -c '  inode=%i nlink=%h %n' "$A2_RKD"/model-0000?-of-00006.safetensors
else
    echo "  NO save — bug?"
fi
