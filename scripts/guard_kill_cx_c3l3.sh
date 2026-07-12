#!/usr/bin/env bash
# guard_kill_cx_c3l3.sh — cx-c3l3 v6-55 is queued in the round's GPU1 worker
# AFTER cx-c3l4 (which we keep). We can't pre-kill it without hitting cx-c3l4,
# so: wait for cx-c3l4-v6 to finish, then kill cx-c3l3's v6-55 the moment it
# launches (uniquely identified by port 8432 — only cx-c3l3 uses it). Killing
# its omk makes run_v6 return, the round wraps, FINISH_ALTW_ROUND_DONE fires,
# and the gated hard-77 rescore proceeds. This script's own cmdline is
# "bash guard_kill_cx_c3l3.sh" (no "8432"), so pgrep never matches itself.
set -uo pipefail
ts(){ date '+%T %Z'; }
C4=/srv/ml/eval_results_lcb_v6/lcb_v6_55/cx-c3l4-v6/summary.json
echo "[guard $(ts)] waiting for cx-c3l4-v6 to finish (keep it), then kill cx-c3l3 ..."
for i in $(seq 1 140); do [ -f "$C4" ] && break; sleep 30; done
[ -f "$C4" ] && echo "[guard $(ts)] cx-c3l4-v6 done" || echo "[guard $(ts)] WARN cx-c3l4-v6 summary missing; proceeding to watch 8432"
for i in $(seq 1 80); do
  pids=$(pgrep -f "8432" 2>/dev/null | tr '\n' ' ')
  if [ -n "${pids// /}" ]; then
    echo "[guard $(ts)] cx-c3l3 v6-55 launched; killing enumerated pids: $pids"
    kill $pids 2>/dev/null; sleep 3
    kill -9 $pids 2>/dev/null
    echo "[guard $(ts)] cx-c3l3 killed"
    break
  fi
  sleep 8
done
echo "###### GUARD_CX_C3L3_DONE $(ts) ######"
