#!/usr/bin/env bash
# relaunch_fast_32k.sh — apply the 32k answer cap to the fast 12B test. Re-derives the current
# fast procs (driver + full_fast48k cells + 8190/8191 servers), kills them by verified PID
# (driver first so it can't auto-advance to pair 2), bumps profile max_tokens 100000->32768,
# sets the partial 100k results aside, relaunches the driver fresh. GPU1 untouched.
set -uo pipefail
WORK=/srv/ml/agentic_loop_12b_test
PROF=$WORK/full_fast48k.yaml
gpu0(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -dc '0-9'; }

echo "=== enumerate current fast procs ==="
DRVS=$(pgrep -f "run_4cell_fast.sh" || true)
CELLS=""
for p in $(pgrep -f agentic_loop_harness); do
  tr "\0" " " < /proc/$p/cmdline 2>/dev/null | grep -q full_fast48k && CELLS="$CELLS $p"
done
SRVS=""
for p in $(pgrep -x llama-server); do
  tr "\0" " " < /proc/$p/cmdline 2>/dev/null | grep -qE "port 819[01]" && SRVS="$SRVS $p"
done
echo "  drivers:$DRVS"; echo "  cells: $CELLS"; echo "  servers:$SRVS"

echo "=== kill driver first, then cells, then servers ==="
[ -n "$DRVS" ]  && kill $DRVS  2>/dev/null && echo "driver killed"
[ -n "$CELLS" ] && kill $CELLS 2>/dev/null && echo "cells killed"
[ -n "$SRVS" ]  && kill $SRVS  2>/dev/null && echo "servers killed"

echo "=== wait GPU0 < 8GB ==="
freed=0
for i in $(seq 1 25); do u=$(gpu0); echo "  t=${i}s GPU0=${u}MiB"; [ "${u:-99999}" -lt 8000 ] && { freed=1; break; }; sleep 1; done
if [ "$freed" -ne 1 ]; then
  for p in $DRVS $CELLS $SRVS; do kill -0 "$p" 2>/dev/null && { echo "KILL9 $p"; kill -9 "$p" 2>/dev/null; }; done
  for i in $(seq 1 15); do u=$(gpu0); [ "${u:-99999}" -lt 8000 ] && { freed=1; break; }; sleep 1; done
fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
[ "$freed" -ne 1 ] && { echo "ABORT: GPU0 not freed"; exit 1; }

echo "=== profile: max_tokens 100000 -> 32768 ==="
sed -i "s/max_tokens: 100000/max_tokens: 32768/" "$PROF"
grep -E "max_tokens|reasoning_budget|parallel|ctx_size|gguf" "$PROF"

echo "=== set partial 100k results aside (clean restart) ==="
for c in embedded reinject_off pr35 pr35_ptfalse; do
  [ -d "$WORK/full_fast48k/$c" ] && mv "$WORK/full_fast48k/$c" "$WORK/full_fast48k/${c}.pre32k" 2>/dev/null || true
done

echo "=== relaunch fast driver (32k) ==="
cd "$WORK"
nohup bash run_4cell_fast.sh > run_fast.driver.log 2>&1 &
echo "driver pid=$!"; disown
sleep 8
head -12 run_fast.driver.log 2>/dev/null
echo "=== booted cells ==="
for p in $(pgrep -f agentic_loop_harness); do
  tr "\0" " " < /proc/$p/cmdline 2>/dev/null | grep -q full_fast48k && echo "  HARNESS $p :: $(tr "\0" " " < /proc/$p/cmdline | grep -oE 'out-dir [^ ]+|port [0-9]+' | tr '\n' ' ')"
done
echo "DONE"
