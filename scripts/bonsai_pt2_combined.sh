#!/usr/bin/env bash
# Combined-corpus arm: mixed calibration corpus + the base model's own traces, same 96 x 1024 windows.
# bonsai_pt2_full.py samples windows from the first 4M chars, so the mix is built to be exactly that long.
set -uo pipefail
ts(){ date -u '+%H:%M:%SZ'; }
PY=/root/anaconda3/envs/omnimergekit/bin/python
S=/mnt/sdc/opencoti-scratch/v6-ternary; W=$S/pt2_combined; mkdir -p "$W/q"
TR=$S/pt2_selftrace/self_traces.txt; MIX=$W/calib_combined.txt
n=$(stat -c%s "$TR"); head -c $((4000000 - n)) /mnt/sdc/ml/omnimerge-v6/calib_train.txt > "$MIX"; cat "$TR" >> "$MIX"
echo "[$(ts)] combined corpus $(stat -c%s "$MIX") B, traces $n B"
cd /srv/ml/scripts && "$PY" -u bonsai_pt2_full.py --model-dir /mnt/sdc/opencoti-models/hf/Qwen/Qwen3.8-27B \
    --prism-gguf /mnt/sdc/opencoti-models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf --calib-text "$MIX" \
    --qdir "$W/q" --device cuda:1 > "$W/run.log" 2>&1
grep -q PT2_FULL_DONE "$W/run.log" || { echo "FATAL ternarization"; tail -5 "$W/run.log"; exit 1; }
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
[ "$free" -ge 75 ] || { echo "[$(ts)] HOLD_EXPORT_DISK: ${free}G free, export needs ~70G; tensors are safe in $W/q"; exit 0; }
PT2_DIR=$W PT2_TAG=pt2mix bash /srv/ml/scripts/bonsai_pt2_export.sh
