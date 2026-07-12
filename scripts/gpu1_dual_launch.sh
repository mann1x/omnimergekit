#!/usr/bin/env bash
# After the F16 LCB retry AND the sub-IQ4 smoke release GPU1, launch 2 concurrent
# quant-sweep workers on GPU1 (distinct ports) — claim-based, they split remaining cells.
set +e
echo "[gpu1-dual] $(date -u) waiting for lcb_retry_driver to finish..."
while pgrep -f lcb_retry_driver.sh >/dev/null 2>&1; do sleep 30; done
echo "[gpu1-dual] $(date -u) LCB retry done; waiting for smoke_subiq4 to finish..."
while pgrep -f smoke_subiq4.sh >/dev/null 2>&1; do sleep 30; done
echo "[gpu1-dual] $(date -u) GPU1 free; launching 2 workers"
cd /srv/ml/scripts
setsid nohup bash quant_sweep_v7.sh 1 8243 >/dev/null 2>&1 < /dev/null & disown
sleep 15
setsid nohup bash quant_sweep_v7.sh 1 8244 >/dev/null 2>&1 < /dev/null & disown
sleep 25
echo "[gpu1-dual] $(date -u) GPU1 workers:"; pgrep -af "quant_sweep_v7.sh 1" | grep -v grep
echo "[gpu1-dual] DONE"
