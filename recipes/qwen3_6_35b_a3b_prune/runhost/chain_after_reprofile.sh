#!/usr/bin/env bash
# Wait for the running re-profile to finish, then deliver BOTH open items unattended:
#   Q1 (#761): rnorm-vs-tc analysis on the freshly populated coder map  -> the HF disc #2 answer
#   R1 (#828): REAM arms C -> B -> E on GPU1                            -> the 2x2 matrix
#
# The wait gates on the real completion signals, not on a timer: the profiler PID being gone
# AND the final map existing. A PID that vanished without writing the map is a FAILURE, not a
# reason to proceed -- the analysis would otherwise silently re-run on the stale checkpoint.
set -uo pipefail

PID=${PID:-1812778}
RECIPE=/srv/ml/repos/omnimergekit/recipes/qwen3_6_35b_a3b_prune
WORK=/mnt/sdc/ream-work
MAP=$WORK/maps/competence_qwen35b_coder_lcbmpe_FULL.json
DROPMAP=$RECIPE/results/drop_map_184e_coder_lcbmpe.json
PY=/srv/ml/envs/envs/omnimergekit/bin/python
LOG=$WORK/chain_after_reprofile.log

exec >>"$LOG" 2>&1
echo "=== chain start $(date -u +%F' '%T) waiting on PID $PID ==="

waited=0
while kill -0 "$PID" 2>/dev/null; do
  sleep 60
  waited=$((waited + 60))
  if [ $((waited % 600)) -eq 0 ]; then
    echo "[$(date -u +%H:%M:%S)] still profiling (${waited}s): $(tail -1 "$WORK/reprofile_full.log")"
  fi
  [ $waited -ge 14400 ] && { echo "ABORT: profiler still alive after 4h"; exit 3; }
done
echo "[$(date -u +%H:%M:%S)] profiler PID $PID exited after ${waited}s"

if [ ! -f "$MAP" ]; then
  echo "ABORT: profiler exited but $MAP was never written -- refusing to analyse the stale"
  echo "checkpoint, which would silently report partial-corpus numbers as the full result."
  tail -20 "$WORK/reprofile_full.log"
  exit 4
fi
echo "map present: $(ls -la "$MAP")"

# ---- Q1: the cheap test promised in the HF discussion -------------------------------------
# Run on the SHIPPED recipe (agg wmax + the targeted_lcb upweight) so the tc<->shipped column
# is a genuine 1.0 control. If it is not ~1.0, the map is not on the shipped basis and the
# rnorm column cannot be attributed to rnorm.
echo; echo "=== Q1: rnorm vs tc on the shipped coder basis ==="
"$PY" "$RECIPE/ream/analyze_rnorm_vs_tc.py" \
  --competence-map "$MAP" --drop-map "$DROPMAP" \
  --agg wmax --cat-weight corpus_targeted_lcb=2.0 \
  --out "$WORK/rnorm_vs_tc_coder.json"
echo ">>> Q1_RC=$?"

# ---- R1: the REAM arms --------------------------------------------------------------------
echo; echo "=== R1: REAM arms C -> B -> E ==="
bash "$RECIPE/ream/run_ream_arms.sh"
echo ">>> R1_RC=$?"

echo "=== chain done $(date -u +%F' '%T) ==="
echo ">>> CHAIN_AFTER_REPROFILE_DONE"
