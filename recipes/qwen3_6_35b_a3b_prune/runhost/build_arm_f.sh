#!/usr/bin/env bash
# Arm F = the rnorm cut. Same pipeline as arm D (inject our saliency, --merging none), with the
# rnorm-scored drop map swapped in for the shipped tc one. That makes F-vs-D a SINGLE-variable
# comparison: which experts the map picked, nothing else. It is the model sfrav's question in
# HF disc #2 asks for -- until it exists, the rnorm result is a selection difference, not a
# quality one.
#
# Two things that must be right or the arm is meaningless:
#  * the map must be the RE-PROFILED one. The shipped coder map carries rnorm as 0.0 everywhere;
#    ranking it would silently order by expert index. make_drop_map.assert_score_populated
#    refuses that, and this script points at the FULL map so the guard never has to fire.
#  * --cat-weight is 1.5/1.5, the RECOVERED shipped recipe -- not the 2.0 written in STATE.md,
#    which does not reproduce the shipped cut (0.8708).
set -uo pipefail

WORK=/mnt/sdc/ream-work
RECIPE=/srv/ml/repos/omnimergekit/recipes/qwen3_6_35b_a3b_prune
BASE=/srv/ml/models/Qwen3.6-35B-A3B
MAP=$WORK/maps/competence_qwen35b_coder_lcbmpe_FULL.json
DROPMAP=$RECIPE/results/drop_map_184e_coder_lcbmpe_rnorm.json
OUT=$WORK/armF_rnorm_nomerge
PY=/srv/ml/envs/envs/omnimergekit/bin/python
LOG=$WORK/build_armF.log

exec >>"$LOG" 2>&1
echo "=== armF build start $(date -u +%F' '%T) ==="

for f in "$MAP" "$DROPMAP" "$BASE/config.json"; do
  [ -s "$f" ] || { echo "ABORT: missing $f"; exit 2; }
done

# Never preempt: GPU1 is ours but a run in flight is not ours to kill.
free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$free" -ge 60000 ] || { echo "ABORT: GPU1 only ${free}MiB free"; exit 3; }
avail=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
[ "$avail" -ge 100 ] || { echo "ABORT: /mnt/sdc only ${avail}G free"; exit 4; }
echo "preflight ok: GPU1 ${free}MiB free, /mnt/sdc ${avail}G free"

[ -d "$OUT" ] && { echo "ABORT: $OUT exists -- refusing to overwrite a built arm"; exit 5; }

cd "$RECIPE/ream"
CUDA_VISIBLE_DEVICES=1 "$PY" omk_ream_merge.py \
  --model "$BASE" --merge-size 184 \
  --saliency reap --merging none \
  --data-root "$WORK" --tokenizer-name qwen36 \
  --save-path "$OUT" --seed 42 \
  --saliency-map "$MAP" --drop-map "$DROPMAP" \
  --score rnorm --agg wmax \
  --cat-weight corpus_targeted_lcb=1.5 --cat-weight corpus_targeted_mpe=1.5
rc=$?
echo "=== armF build end rc=$rc $(date -u +%F' '%T) ==="

# Gate on the sentinel, not on rc -- a crashed builder can still exit 0 through a pipe.
if grep -q ">>> OMK_REAM_DONE" "$LOG"; then
  echo ">>> ARMF_BUILT_OK"
else
  echo ">>> ARMF_BUILD_FAIL (no OMK_REAM_DONE sentinel)"
fi
