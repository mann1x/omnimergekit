#!/usr/bin/env bash
# Wait for the a3b-gepo3 build to produce a REAL Q6_K, then run the 10-bench suite.
#
# Gates on the ARTIFACT, not on the process: build_a3b_gepo3.sh can exit non-zero at a
# late gate having still written a partial file, and a dead process with no GGUF looks
# identical to a slow one from the outside. The predicate here is "the sentinel is in
# the log AND the Q6_K exists AND it starts with the GGUF magic".
#
# Bracketed pgrep ([b]uild_) so the pattern cannot match this script's own command line
# -- the bs2 self-match that returns 255.
set -uo pipefail

W=/mnt/sdc/ml/brevity/gepo
LOG=$W/chain_gepo3.log
Q6=$W/gguf_gepo3/a3b-gepo3-Q6_K.gguf
DEADLINE=$(( $(date +%s) + 6*3600 ))

say(){ echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG"; }

say "=== chain_gepo3: waiting for the build ==="
while :; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    say "ABORT: 6h deadline passed, build has not produced $Q6"; exit 1
  fi
  done_marker=$(grep -ac "A3B_GEPO3_BUILD_DONE" "$W/build_a3b_gepo3.log" 2>/dev/null || echo 0)
  alive=$(pgrep -f "[b]uild_a3b_gepo3.sh" | wc -l)
  if [ "$done_marker" -gt 0 ]; then
    say "build sentinel present"; break
  fi
  if [ "$alive" -eq 0 ]; then
    sleep 60
    done_marker=$(grep -ac "A3B_GEPO3_BUILD_DONE" "$W/build_a3b_gepo3.log" 2>/dev/null || echo 0)
    if [ "$done_marker" -eq 0 ]; then
      say "REFUSE: build process gone with no A3B_GEPO3_BUILD_DONE -- it died."
      tail -n 20 "$W/build_a3b_gepo3.log" | tee -a "$LOG"
      exit 2
    fi
  fi
  sleep 60
done

# artifact gate -- the sentinel alone is not the file
if [ ! -s "$Q6" ] || [ "$(head -c4 "$Q6")" != "GGUF" ]; then
  say "REFUSE: $Q6 missing or not a GGUF despite the sentinel"; exit 3
fi
say "Q6_K present ($(stat -c %s "$Q6" | numfmt --to=iec))"

# GPU1 must actually be free -- the build's imatrix leg holds it
for i in $(seq 1 60); do
  free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "${free:-0}" -ge 60000 ] && break
  sleep 30
done
say "GPU1 free=${free}MiB"

say "LAUNCH suite_gepo3.sh (10 benches, sampler=recommended, ~4-5h)"
bash "$W/suite_gepo3.sh"
rc=$?
say "suite_gepo3.sh rc=$rc"
say "CHAIN_GEPO3_EXIT $rc"
