#!/usr/bin/env bash
# pod_v10_suite_rerun.sh <map> <gpu> <port> — ON THE POD: wait for the map's queue session (v10q_<short>) to end,
# keep its queue log as q_<map>.run<N>.log, then rerun the queue: scored cells are skipped, so only the cells
# that failed run again. 2026-10-17: hub_k12 humanevalplus_full died at startup (rc 20, CUDA0 14.9 GB free) because
# the hub_k9 Q6_K sanity server was briefly on GPU0 — a collision, not a model result.
set -uo pipefail
n=$1; g=$2; p=$3; S=/workspace/v10/suite/logs; sess=v10q_${n#hub_}
say(){ echo "[rerun $n $(date -u +%FT%TZ)] $*"; }
while tmux has-session -t "$sess" 2>/dev/null; do sleep 60; done
k=1; while [ -e "$S/q_$n.run$k.log" ]; do k=$((k+1)); done
cp "$S/q_$n.log" "$S/q_$n.run$k.log"; say "queue $sess ended; its log kept as q_$n.run$k.log; rerunning failed cells"
bash /workspace/omnimergekit/scripts/pod_v10_suite.sh queue "$n" "$g" "$p"
