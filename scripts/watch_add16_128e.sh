#!/usr/bin/env bash
# watch_add16_128e.sh — report-only: surface ADD16-Q3_K_S + 128e-Q3_K_S gate results as the
# orchestrator produces them, then exit when the pipeline finishes. No kills, no writes to models.
set -uo pipefail
LOG=/mnt/sdc/ml/t223_fk/add16_128e_gates.log
AL=/srv/ml/agentic_loop
PY=/root/anaconda3/envs/omnimergekit/bin/python
ts(){ date '+%T %Z'; }
seen_add16=0; seen_128e=0
emit(){ # $1 results json
  "$PY" - "$1" 2>/dev/null <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    for r in d["results"]:
        loops=r.get("loops", round(r.get("loop_rate",0)*r["seeds"]))
        print("    %-12s fails=%d/%d loops=%s fail_rate=%.1f%%"%(r["config"],r["fails"],r["seeds"],loops,100*r["fail_rate"]))
except Exception as e:
    print("    (summary not yet parseable: %s)"%e)
PY
}
for i in $(seq 1 160); do   # ~4h ceiling
  if [ $seen_add16 = 0 ] && grep -q "=== ADD16-Q3_K_S DONE ===" "$LOG" 2>/dev/null; then
    echo "[watch $(ts)] ===== ADD16-Q3_K_S gate DONE (vendor_minp_rep) — vs STD16 Q3_K_S 19/16 ====="
    grep -E "FAIL=True" "$AL/logs/minp48_replay_ADD16-Q3_K_S.log" 2>/dev/null | head -5 || true
    emit "$AL/results/ADD16-Q3_K_S.json"; seen_add16=1
  fi
  if [ $seen_128e = 0 ] && grep -q "=== 128e-Q3_K_S-vbase DONE ===" "$LOG" 2>/dev/null; then
    echo "[watch $(ts)] ===== 128e-Q3_K_S gate DONE (google/vendor_base temp 1.0) — base reference ====="
    emit "$AL/results/128e-Q3_K_S-vbase.json"; seen_128e=1
  fi
  if grep -q "orchestrator DONE" "$LOG" 2>/dev/null; then echo "[watch $(ts)] ===== PIPELINE DONE ====="; break; fi
  sleep 90
done
echo "ADD16_128E_WATCH_DONE"
