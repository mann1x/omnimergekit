#!/usr/bin/env bash
# Wait for the in-flight orphaned multipl_e_100 cell to finish, then run the
# full AC suite. The orphan is a VALID cell (same GGUF, same sampler args, MPE
# does not use lm-eval so it was unaffected by the PATH defect) -- killing it
# would throw away real work, and its lock correctly blocks a concurrent start.
set -uo pipefail
ts(){ date -u +%H:%M:%S; }
say(){ echo "[$(ts)Z] $*"; }
TARGET=${1:?pid of the orphan omk_eval}

say "waiting on orphan pid $TARGET (multipl_e_100)"
for i in $(seq 1 720); do          # up to 2h
  kill -0 "$TARGET" 2>/dev/null || break
  sleep 10
done
if kill -0 "$TARGET" 2>/dev/null; then
  say "ABORT: orphan $TARGET still alive after 2h - not forcing"
  exit 1
fi
say "orphan gone; MPE artifacts:"
ls -la /srv/ml/eval_results/v6_ac/multipl_e_100/*/summary.json 2>/dev/null || say "  (no MPE summary - suite will redo it)"
sleep 15                            # let the server port release
say "launching full AC suite"
exec bash /srv/ml/scripts/v6_ac_eval.sh
