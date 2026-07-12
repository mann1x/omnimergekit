#!/bin/bash
# Run on bs2: atomic-restore A2 to pristine pre-EAC + launch iterate_a2.sh fresh
# Author: claude opus 4.7  2026-05-29
set -uo pipefail

A2=/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it
A2_EAC=/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-it
A2_EAC_OLD=/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-it_pre_fix_${TS:-$(date +%Y%m%d_%H%M%S)}

echo "=== STEP 1: preserve old A2_EAC (rename, don't delete — backups inside) ==="
if [ -d "$A2_EAC" ]; then
    mv "$A2_EAC" "$A2_EAC_OLD"
    echo "  A2_EAC → $A2_EAC_OLD"
else
    echo "  A2_EAC does not exist — skip"
    A2_EAC_OLD=""
fi

echo
echo "=== STEP 2: identify .pre_eac_calibrate restore source ==="
if [ -n "$A2_EAC_OLD" ] && ls "$A2_EAC_OLD"/*.pre_eac_calibrate >/dev/null 2>&1; then
    RESTORE_SRC="$A2_EAC_OLD"
else
    # Fall back to /srv/ml/models/variants
    VS=/srv/ml/models/variants/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-it
    if ls "$VS"/*.pre_eac_calibrate >/dev/null 2>&1; then
        RESTORE_SRC="$VS"
    else
        echo "  FATAL: no .pre_eac_calibrate source found" >&2
        exit 2
    fi
fi
echo "  source: $RESTORE_SRC"

echo
echo "=== STEP 3: atomic-restore A2 shards (break hardlinks) ==="
for i in 1 2 3 4 5 6; do
    shard="model-0000${i}-of-00006.safetensors"
    src="$RESTORE_SRC/${shard}.pre_eac_calibrate"
    dst="$A2/${shard}"
    if [ ! -f "$src" ]; then
        echo "  WARN: $src missing — skip shard $i"
        continue
    fi
    # nlink BEFORE
    nl_before=$(stat -c '%h' "$dst")
    ino_before=$(stat -c '%i' "$dst")
    # cp to .tmp (new inode) + mv to break hardlink + restore content
    cp "$src" "${dst}.restore_tmp"
    mv "${dst}.restore_tmp" "$dst"
    nl_after=$(stat -c '%h' "$dst")
    ino_after=$(stat -c '%i' "$dst")
    echo "  shard $i  inode $ino_before (nlink $nl_before) → $ino_after (nlink $nl_after)"
done

echo
echo "=== STEP 4: verify A2 is now isolated (different inodes from siblings) ==="
echo "--- A2 main shards ---"
for s in "$A2"/model-*.safetensors; do
    stat -c '  inode=%i nlink=%h %n' "$s"
done
echo "--- A2_EAC_OLD main shards (should be DIFFERENT inodes than A2 now) ---"
[ -n "$A2_EAC_OLD" ] && for s in "$A2_EAC_OLD"/model-0000?-of-00006.safetensors; do
    stat -c '  inode=%i nlink=%h %n' "$s"
done

echo
echo "=== STEP 5: launch iterate_a2.sh (will cp -a A2 → A2_EAC, run EAC + KD) ==="
TS_LAUNCH=$(date +%Y%m%d_%H%M%S)
LOG=/srv/ml/logs/iterate_a2_post_fix_$TS_LAUNCH.log
mkdir -p /srv/ml/logs
nohup bash /srv/ml/scripts/iterate_a2.sh > "$LOG" 2>&1 &
PID=$!
disown $PID 2>/dev/null || true
echo "  iterate_a2.sh PID=$PID"
echo "  LOG=$LOG"
echo "  expected: cp -a (5 min) + EAC 150 steps (~10 min) + KD 100 steps (~10 min) = ~25 min"
echo "$PID" > /tmp/iterate_a2_pid
echo "$LOG" > /tmp/iterate_a2_log
echo
sleep 10
echo "=== T+10s ==="
head -25 "$LOG"
ps -p $PID -o pid,etime,cmd 2>/dev/null | tail -3 || echo "DEAD"
