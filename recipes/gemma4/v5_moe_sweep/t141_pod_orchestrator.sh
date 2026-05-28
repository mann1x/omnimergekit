#!/bin/bash
# t141_pod_orchestrator.sh — runs ON THE POD inside tmux. Takes over from
# the (defunct) solidpc-side master after recreation+launch already happened.
#
# Preconditions (verified at launch):
#   /srv/ml/logs/t141/ablation.pid   — PID of run_shared_x_pes_ablation.sh
#   /srv/ml/logs/t141/eac.pid        — PID of run_eac_baseline.sh
#   GPU 0 = ablation, GPU 1 = EAC (per launch convention)
#
# Phases:
#   3. Wait for ablation + EAC to finish (poll PIDs every 5 min)
#   4. Snapshot scores (read summary.json, never raw exact_match)
#   5. Launch Router-KD if --with-kd                       (3-4h on 2 GPUs)
#   6. Wait Router-KD finish
#   7. Print final result block (rsync-back is solidpc-pull, not pushed here)
#
# Usage (intended): launch inside tmux on pod:
#   ssh linode-blackswan-2 'tmux new -d -s t141 "bash /srv/ml/scripts/t141_pod_orchestrator.sh --with-kd"'
#
# Resumable: re-launching after a partial run picks back up — PID polls
# treat already-dead PIDs as "done" and Router-KD honors its own resume.

set -uo pipefail

WITH_KD=0
SKIP_KD_IF_FAIL=1   # if any ablation cell or EAC failed, don't waste 4h on KD
while [ $# -gt 0 ]; do
  case "$1" in
    --with-kd)         WITH_KD=1; shift;;
    --no-kd-on-fail)   SKIP_KD_IF_FAIL=1; shift;;
    --force-kd)        SKIP_KD_IF_FAIL=0; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

BM=/srv/ml
LOG_DIR=$BM/logs/t141
RES=$BM/eval_results_t141_shared_x_pes
TS=$(date +%Y%m%d_%H%M%S)
ORCH_LOG=$LOG_DIR/orchestrator_${TS}.log
PY=/root/anaconda3/envs/omnimergekit/bin/python

mkdir -p "$LOG_DIR"

# tee everything to a log so tmux scrollback isn't the only record
exec > >(tee -a "$ORCH_LOG") 2>&1

ts() { date -u +%FT%TZ; }
log() { printf "[%s] %s\n" "$(ts)" "$*"; }

log "=== T141 pod orchestrator starting ==="
log "  with-kd=$WITH_KD  skip-kd-if-fail=$SKIP_KD_IF_FAIL"
log "  log: $ORCH_LOG"

# ---------- preconditions ----------
for f in ablation.pid eac.pid; do
  if ! [ -f "$LOG_DIR/$f" ]; then
    log "ERROR: $LOG_DIR/$f missing — ablation/EAC not yet launched. Bailing."
    exit 3
  fi
done

log "  ablation PID file: $LOG_DIR/ablation.pid"
log "  EAC      PID file: $LOG_DIR/eac.pid"
log "  (re-reading PID files every poll so relaunches transparently take over)"

# ---------- phase 3: wait both ----------
log "phase 3 — waiting for ablation + EAC"
prev_abl_pid=""
prev_eac_pid=""
while true; do
  ABL_PID=$(cat $LOG_DIR/ablation.pid 2>/dev/null || echo "")
  EAC_PID=$(cat $LOG_DIR/eac.pid 2>/dev/null || echo "")
  abl_run=N; eac_run=N
  [ -n "$ABL_PID" ] && kill -0 "$ABL_PID" 2>/dev/null && abl_run=Y
  [ -n "$EAC_PID" ] && kill -0 "$EAC_PID" 2>/dev/null && eac_run=Y
  # log when a PID swap happens (relaunch detection)
  if [ "$ABL_PID" != "$prev_abl_pid" ]; then
    log "  [pid-swap] ablation PID is now $ABL_PID (was '$prev_abl_pid')"
    prev_abl_pid=$ABL_PID
  fi
  if [ "$EAC_PID" != "$prev_eac_pid" ]; then
    log "  [pid-swap] EAC PID is now $EAC_PID (was '$prev_eac_pid')"
    prev_eac_pid=$EAC_PID
  fi
  # GPU snapshot
  gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
        | awk -F',' 'NR<=2{printf "gpu%s=%s/%s  ",$1+0,$3+0,$2+0}END{print ""}')
  log "  ablation=$abl_run(pid=$ABL_PID)  EAC=$eac_run(pid=$EAC_PID)  $gpu"
  if [ "$abl_run" = "N" ] && [ "$eac_run" = "N" ]; then
    log "  both jobs finished"
    break
  fi
  sleep 300
done

# ---------- phase 4: snapshot scores ----------
log "phase 4 — scoring snapshot (summary.json, never raw exact_match)"
ABL_FAILS=0
ABL_PASSES=0
for cell in A1 A2 B1 B2; do
  for tpl in humanevalplus_full multipl_e_100; do
    s="$RES/$tpl/t141-62e-$cell/summary.json"
    if [ -f "$s" ]; then
      score=$("$PY" -c "import json,sys
d=json.load(open('$s'))
print(round(d.get('score',0)*100,2))" 2>/dev/null || echo "ERR")
      log "  $cell  $tpl  ->  $score%"
      ABL_PASSES=$((ABL_PASSES+1))
    else
      log "  $cell  $tpl  ->  MISSING"
      ABL_FAILS=$((ABL_FAILS+1))
    fi
  done
done

# EAC eval (if any)
EAC_FAIL=0
for s in "$BM/eval_results_eac_baseline"/*/summary.json; do
  if [ -f "$s" ]; then
    name=$(basename $(dirname "$s"))
    score=$("$PY" -c "import json;print(round(json.load(open('$s')).get('score',0)*100,2))" 2>/dev/null || echo "ERR")
    log "  EAC $name -> $score%"
  fi
done
# crude EAC pass test — succeeded if its DONE marker exists
if [ ! -f "$LOG_DIR/eac_DONE" ]; then
  log "  EAC: no DONE marker (may be expected — depends on run_eac_baseline.sh contract)"
fi

log "  ablation cells with summary: $ABL_PASSES / 8 (4 cells × 2 templates)"

# ---------- phase 5: Router-KD ----------
KD_LAUNCHED=0
if [ "$WITH_KD" -eq 1 ]; then
  if [ "$SKIP_KD_IF_FAIL" -eq 1 ] && [ "$ABL_FAILS" -gt 4 ]; then
    log "phase 5 — SKIP Router-KD: $ABL_FAILS/8 ablation cells failed (>50%); 3-4h KD not justified"
  else
    log "phase 5 — launching Router-KD on both GPUs (3-4h wall-clock)"
    KD_LOG=$LOG_DIR/router_kd_${TS}.log
    nohup bash $BM/scripts/run_router_kd.sh --run > "$KD_LOG" 2>&1 &
    KD_PID=$!
    disown
    echo "$KD_PID" > $LOG_DIR/router_kd.pid
    log "  Router-KD PID $KD_PID — log $KD_LOG"
    KD_LAUNCHED=1

    log "phase 6 — waiting Router-KD"
    while true; do
      if ! kill -0 "$KD_PID" 2>/dev/null; then
        log "  Router-KD finished"
        break
      fi
      gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
            | awk -F',' 'NR<=2{printf "gpu%s=%s/%s  ",$1+0,$3+0,$2+0}END{print ""}')
      log "  Router-KD still running  $gpu"
      sleep 600
    done
  fi
fi

# ---------- phase 7: final summary ----------
log "=== T141 final summary ==="
log "  ablation results : $RES/"
log "  EAC results      : $BM/eval_results_eac_baseline/"
[ "$KD_LAUNCHED" -eq 1 ] && log "  Router-KD results: $BM/eval_results_router_kd/"
log "  rsync-back to solidpc: not auto-pushed (solidpc has no inbound SSH);"
log "    on solidpc:  rsync -ah linode-blackswan-2:$RES/ \\"
log "                       /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/eval_results_t141_shared_x_pes/"

touch $LOG_DIR/orchestrator_DONE
log "=== orchestrator DONE — see $ORCH_LOG ==="
