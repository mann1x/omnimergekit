#!/usr/bin/env bash
# stop_nocap_launch_fast.sh — stop the 2 no-cap GPU0 cells by VERIFIED PID, wait for GPU0 VRAM
# to free, then launch the fast (reasoning_budget=48000) 4-cell driver on GPU0. GPU1 untouched.
set -uo pipefail
DRV=(1683713 1683714)   # embedded, reinject_off harness drivers (verified)
SRV=(1683718 1683717)   # port 8190, 8191 llama-servers (verified)
gpu0(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -dc '0-9'; }

echo "=== SIGTERM no-cap GPU0 cells (drivers first, then servers) ==="
kill "${DRV[@]}" 2>/dev/null; echo "drivers TERM rc=$?"
kill "${SRV[@]}" 2>/dev/null; echo "servers TERM rc=$?"

echo "=== wait for GPU0 VRAM < 8GB ==="
freed=0
for i in $(seq 1 25); do
  u=$(gpu0); echo "  t=${i}s GPU0=${u}MiB"
  [ "${u:-99999}" -lt 8000 ] && { freed=1; break; }
  sleep 1
done
if [ "$freed" -ne 1 ]; then
  echo "=== not freed gracefully; SIGKILL stragglers ==="
  for p in "${DRV[@]}" "${SRV[@]}"; do kill -0 "$p" 2>/dev/null && { echo "KILL9 $p"; kill -9 "$p" 2>/dev/null; }; done
  for i in $(seq 1 15); do u=$(gpu0); [ "${u:-99999}" -lt 8000 ] && { freed=1; break; }; sleep 1; done
fi
echo "=== final pid status + GPU0 ==="
for p in "${DRV[@]}" "${SRV[@]}"; do kill -0 "$p" 2>/dev/null && echo "  $p ALIVE" || echo "  $p gone"; done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
[ "$freed" -ne 1 ] && { echo "ABORT: GPU0 not freed — NOT launching"; exit 1; }

echo "=== launch fast driver (nohup) ==="
cd /srv/ml/agentic_loop_12b_test
nohup bash run_4cell_fast.sh > run_fast.driver.log 2>&1 &
echo "driver pid=$!"; disown
sleep 8
echo "=== driver log head ==="; head -20 run_fast.driver.log 2>/dev/null
echo "=== new fast cells booting (GPU0) ==="
for p in $(pgrep -f agentic_loop_harness); do
  a=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null)
  echo "$a" | grep -q full_fast48k && echo "  HARNESS $p :: $(echo "$a" | grep -oE "out-dir [^ ]+|port [0-9]+" | tr '\n' ' ')"
done
for p in $(pgrep -x llama-server); do echo "  SERVER $p :: $(tr "\0" " " < /proc/$p/cmdline 2>/dev/null | grep -oE 'port [0-9]+')"; done
echo "DONE"
