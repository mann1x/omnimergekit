#!/usr/bin/env bash
# pod_v10_after_q.sh — ON THE POD: follow-on bench queue on one GPU, started after that GPU's running queue ends
# (marker line in its log), so a second llama-server never shares the GPU. Owner 2026-10-18 13:39: "run ifeval too on
# the same builds, those which doesn't have already the score" -> ifeval_100 on a_ref / a_ml3 / a_core5 (hub_k12 0.94,
# hub_k9 0.95 already scored on the same basis 10-17). Each map: pod_v10_suite_q.sh <map> <gpu> <port> <bench>
# (release basis: Q6_K, greedy_pen live-gated, q8 KV, eval tree b9700+PEG; a scored summary is skipped, never redone).
# usage: pod_v10_after_q.sh <gpu> <first_port> <wait_log> <wait_marker> <bench> <map>...
set -uo pipefail
G=$1; P=$2; WL=$3; WM=$4; B=$5; shift 5
W=/workspace; L=$W/v10/lcb77/logs; S=$W/omnimergekit/scripts
say(){ echo "[after GPU$G $(date -u +%FT%TZ)] $*"; }
say "waiting for '$WM' in $WL"
for _ in $(seq 1 2880); do grep -q "$WM" "$WL" 2>/dev/null && break; sleep 30; done
grep -q "$WM" "$WL" 2>/dev/null || { say "AFTERQ_FAIL '$WM' never appeared in $WL (24 h)"; exit 1; }
sleep 60
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -dc 0-9)
[ "$used" -lt 2000 ] || { say "AFTERQ_FAIL GPU$G still has ${used} MiB in use after '$WM'"; exit 1; }
fails=0
for n in "$@"; do
  say "$n: $B (port $P)"
  bash $S/pod_v10_suite_q.sh "$n" "$G" "$P" "$B" > $L/q_${B}_$n.log 2>&1
  grep -h -E "RESULT|CELL_|Q_FAIL|QUEUE_DONE|skip" $L/q_${B}_$n.log | sed "s/^/  /"
  grep -q "QUEUE_DONE fails=0" $L/q_${B}_$n.log || fails=$((fails+1))
  P=$((P+1))
done
say "AFTERQ_DONE $B fails=$fails"
