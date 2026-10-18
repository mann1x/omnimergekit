#!/usr/bin/env bash
# pod_v10_lcb77_q.sh — ON THE POD: one GPU's LCB-77 queue over v9-agentic maps; each map waits for its READY_<map>
# (pod_v10_lcb77.sh / solidpc upload), then runs pod_v10_suite_q.sh <map> <gpu> <port> lcb_v6_77q (release basis).
# WAIT_FOR=<marker>: wait for $M/<marker> before the first map (GPU1 waits for READY_a_core5 so its first llama-server
# never shares GPU1 with a build's Q6_K sanity server — the 2026-10-17 rc=20 startup collision).
# usage: [WAIT_FOR=<marker>] pod_v10_lcb77_q.sh <gpu> <first_port> <map>...
set -uo pipefail
G=$1; P=$2; shift 2
W=/workspace; M=$W/v10/lcb77; L=$M/logs; S=$W/omnimergekit/scripts
say(){ echo "[lcbq GPU$G $(date -u +%FT%TZ)] $*"; }
fails=0
if [ -n "${WAIT_FOR:-}" ]; then
  say "waiting for $WAIT_FOR"
  for _ in $(seq 1 1440); do [ -f $M/$WAIT_FOR ] && break; [ -f $M/FAIL ] && break; sleep 30; done
  [ -f $M/$WAIT_FOR ] || say "WARN $WAIT_FOR never appeared (build FAIL or timeout); maps gate on their own READY"
fi
for n in "$@"; do
  ok=0
  for _ in $(seq 1 1440); do
    [ -f $M/READY_$n ] && { ok=1; break; }
    { [ -f $M/FAIL ] || [ -f $M/UPLOAD_FAIL ]; } && break
    sleep 30
  done
  [ $ok = 1 ] || { say "MAP_FAIL $n: never READY (build/upload failed or 12 h timeout)"; fails=$((fails+1)); continue; }
  say "$n: READY -> lcb_v6_77q (port $P)"
  bash $S/pod_v10_suite_q.sh "$n" "$G" "$P" lcb_v6_77q > $L/q_$n.log 2>&1
  grep -h -E "RESULT|CELL_|Q_FAIL|QUEUE_DONE" $L/q_$n.log | sed "s/^/  /"
  grep -q "QUEUE_DONE fails=0" $L/q_$n.log || fails=$((fails+1))
  P=$((P+1))
done
say "LCBQ_DONE fails=$fails"
