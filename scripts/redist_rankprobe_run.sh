#!/usr/bin/env bash
# T191 E-RankProbe orchestrator (the LEAD diffuse-multilingual capacity test):
#   capture(expert_kd, router_in tap; DISJOINT multilingual calib) -> single-layer
#   trainable rank probe -> held-out block-output divergence verdict.
# The capture corpus MUST be disjoint from any test the verdict is judged against;
# this is a CAPACITY probe (held-out divergence), not a loop_screen.
# Usage: redist_rankprobe_run.sh [gpu] [layers] [max_seqs]
set -uo pipefail
GPU="${1:-0}"; LAYERS="${2:-5,12,18,25}"; MAXSEQ="${3:-120}"
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SCR=/srv/ml/scripts
TEACHER=/srv/ml/models/base/gemma-4-26B-A4B-it
STUDENT=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it
KM=/srv/ml/scripts/a2_keep_metadata.json
CORPUS=/mnt/sdc/ml/corpora/redist_calib_multilingual.jsonl
WORK=/srv/ml/redist_work
CAP="$WORK/capture_multilingual_expert_kd.pt"
mkdir -p "$WORK"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L(){ echo "[rankprobe-run $(date -u +%H:%M:%S)] $*"; }

L "=== [1/2] capture (expert_kd: router_in+swiglu_in+block_out; corpus=$(basename "$CORPUS") seqs=$MAXSEQ) ==="
"$PY" "$SCR/redist.py" capture --driver multilingual --method expert_kd --teacher "$TEACHER" \
  --corpus "$CORPUS" --keep-meta "$KM" --max-seqs "$MAXSEQ" --max-tokens 512 \
  --device cuda:0 --workdir "$WORK" --scripts-dir "$SCR" || { L "CAPTURE FAIL"; exit 1; }

L "=== [2/2] rank probe (layers=$LAYERS, 400 steps, held-out tail 20%) ==="
"$PY" "$SCR/redist_rank_probe.py" --student "$STUDENT" --capture "$CAP" \
  --layers "$LAYERS" --steps 400 --lr 1e-3 --heldout 0.2 \
  --device cuda:0 --out "$WORK/rankprobe_multilingual.json" || { L "PROBE FAIL"; exit 1; }
L "RANKPROBE_RUN_DONE"
