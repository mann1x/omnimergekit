#!/bin/bash
# iterate_a2.sh — Council-recommended Router-KD iterate on A2.
# Phase A: EAC-MoE warmup (150 steps, LR 1e-3) to align top-K BEFORE KD.
# Phase B: Router-KD with reduced LR 1e-5 + max-steps 100 (council recipe).
# Original failed run used paper recipe (LR 5e-5, 375 steps) → off-manifold sink.
set -uo pipefail
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=$BM/logs/router_kd_iterate_A2_$TS
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/chain.log
exec > >(tee "$LOG") 2>&1

TEACHER=$BM/google/gemma-4-26B-A4B-it
A2=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it
A2_EAC=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-it
A2_RKD=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-rkd-it
KD_CORPUS=$BM/scripts/router_calib_corpus.jsonl
EAC_CORPUS=$BM/scripts/eac_corpus_wiki2_plus_calib.txt

echo "[$(date -Iseconds)] === RKD-iterate A2 (council recipe) ==="
echo "  teacher  : $TEACHER"
echo "  student  : $A2"
echo "  eac out  : $A2_EAC"
echo "  rkd out  : $A2_RKD"
echo "  EAC LR/steps : 1e-3 / 150 (council 120-200)"
echo "  KD  LR/steps : 1e-5 / 100 (council recipe; paper was 5e-5/375)"
echo

# Phase A: EAC warmup (in-place on -eac-it copy)
echo "[$(date -Iseconds)] === Phase A: EAC warmup ==="
if [ ! -d "$A2_EAC" ]; then
    echo "  [setup] hardlink-copy $A2 -> $A2_EAC"
    cp -a "$A2" "$A2_EAC"  # was cp -al — hardlink dedup bug fix 2026-05-29
fi

$PY $BM/scripts/router_eac_calibrate.py \
    --phase both \
    --base-dir "$TEACHER" \
    --variant-dir "$A2_EAC" \
    --drop-map /srv/ml/repos/omnimergekit/scripts/v6coder_C6v3lcb_62e_fc15_25_p8_drop_map.json \
    --corpus-file "$EAC_CORPUS" \
    --n-seq 128 --seq-len 2048 --batch-size 4 \
    --calib-k 16 --lr 1e-3 --steps 150 \
    --max-gpu-gib 80 --max-cpu-gib 400 \
    --cache-dir $BM/eac_cache_A2_iterate \
    2>&1
RC_EAC=$?
if [ $RC_EAC -ne 0 ]; then
    echo "[$(date -Iseconds)] FATAL: EAC phase failed exit=$RC_EAC"
    exit $RC_EAC
fi

# Phase B: Router-KD with reduced LR + max-steps
echo
echo "[$(date -Iseconds)] === Phase B: Router-KD (LR 1e-5, max-steps 100) ==="
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
    --canary-gate \
    2>&1
RC_KD=$?

echo
echo "[$(date -Iseconds)] === Iterate complete ==="
echo "  EAC exit: $RC_EAC"
echo "  KD  exit: $RC_KD"
if [ -d "$A2_RKD" ]; then
    echo "  Wrote: $A2_RKD"
    ls "$A2_RKD" | head -5
else
    echo "  NO save (canary gate FAILED again — checkpoint at $LOG_DIR/ckpt)"
fi
