#!/bin/bash
# Watcher: waits for the current matrix to exit, then launches the
# wiki2+extended follow-up chain. Reads matrix PID from /tmp/matrix_resume_pid.
# Author: claude opus 4.7  2026-05-29
set -u
MATRIX_PID=$(cat /tmp/matrix_resume_pid 2>/dev/null || echo 0)
TS=$(date +%Y%m%d_%H%M%S)
WLOG=/srv/ml/logs/wiki2_followup_watcher_${TS}.log
exec > "$WLOG" 2>&1

echo "[$(date -Iseconds)] watcher start  matrix_pid=$MATRIX_PID"
if [ "$MATRIX_PID" = 0 ]; then
    echo "  no matrix PID in /tmp/matrix_resume_pid — exiting"; exit 1
fi
if ! kill -0 "$MATRIX_PID" 2>/dev/null; then
    echo "  matrix PID $MATRIX_PID already dead — proceeding to launch follow-up"
else
    echo "  matrix PID $MATRIX_PID is alive, polling every 60s"
    while kill -0 "$MATRIX_PID" 2>/dev/null; do
        sleep 60
    done
    echo "[$(date -Iseconds)] matrix PID $MATRIX_PID exited"
fi

MATRIX_LOG=$(cat /tmp/matrix_resume_log 2>/dev/null || echo)
if [ -n "$MATRIX_LOG" ] && [ -f "$MATRIX_LOG" ]; then
    echo "[$(date -Iseconds)] matrix log: $MATRIX_LOG"
    if grep -q 'matrix chain complete\|=== post-Track5.*complete' "$MATRIX_LOG" 2>/dev/null; then
        echo "  matrix completed normally"
    else
        echo "  matrix exited without 'complete' marker — likely crashed or partial"
        echo "  last 5 lines of matrix log:"
        tail -5 "$MATRIX_LOG" 2>/dev/null
    fi
fi

df -h / | tail -1
free_kb=$(df -k / | awk 'NR==2 {print $4}')
free_gb=$((free_kb / 1024 / 1024))
echo "[$(date -Iseconds)] free disk: ${free_gb}G"
if [ "$free_gb" -lt 200 ]; then
    echo "  WARNING: less than 200G free, follow-up will fail. NOT auto-launching."
    exit 1
fi

FLOG=/srv/ml/logs/wiki2_followup_launch_${TS}.log
echo "[$(date -Iseconds)] launching wiki2 follow-up chain, log: $FLOG"
nohup bash /srv/ml/scripts/post_matrix_wiki2_followup.sh > "$FLOG" 2>&1 < /dev/null &
FPID=$!
disown $FPID 2>/dev/null || true
echo "$FPID" > /tmp/wiki2_followup_pid
echo "$FLOG" > /tmp/wiki2_followup_log
echo "  follow-up PID=$FPID"
sleep 5
ps -p $FPID -o pid,etime,cmd 2>&1 | tail -3
echo "[$(date -Iseconds)] watcher done"
