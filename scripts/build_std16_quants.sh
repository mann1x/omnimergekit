#!/usr/bin/env bash
# build_std16_quants.sh — CPU build-ahead of the STD16 GGUF cohort, imatrix on EVERY tier
# (user directive 2026-06-22). --no-upload => LOCAL GGUFs only; HF/ollama publish is a
# SEPARATE, gated stage (after the 48-seed loop gate + HE+/MPE100 + user GO).
#
# - imatrix-everywhere via qg_imatrix_all.py (clears IMATRIX_EXCLUDE).
# - Reuses pre-staged F16 + Q6_K + imatrix.dat already in the cohort dir (no recompute).
# - Disk-bounded: stops at FLOOR_GB so it never fills /mnt/sdc; remaining tiers build after
#   validate+publish+delete frees space. Resumable via .done markers. Per-tier CPU sanity.
# - CPU-only (--ngl 0): coexists with the GPU-bound 12B harness. QAT lane is separate.
set -uo pipefail
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH
PY=/srv/ml/envs/envs/omnimergekit/bin/python
WRAP=/srv/ml/scripts/qg_imatrix_all.py
WORK=/mnt/sdc/ml/std16_cohort
GGUFDIR=$WORK/gemma-4-A4B-98e-v7-coder-it-GGUF
MODEL=$WORK/gemma-4-A4B-98e-v7-coder-it
BASEID=ManniX-ITA/gemma-4-A4B-98e-v7-coder-it
DONE=$WORK/build_done; mkdir -p "$DONE"
FLOOR_GB=55
export OMK_NO_README=1
ts(){ date '+%T %Z'; }
freeGB(){ df -BG --output=avail "$GGUFDIR" | tail -1 | tr -dc '0-9'; }

# High->low priority: loop-safe K/Q tiers first; loop-risky low + CD last (so if a low tier
# loops with the reused imatrix and needs a balanced rebuild, little build work is wasted).
TIERS=(Q8_0 Q6_K_L Q5_K_L Q5_K_S Q4_K_L Q4_K_M Q4_K_S Q4_1 Q4_0 \
       Q3_K_XL Q3_K_L Q3_K_M Q3_K_S IQ4_NL IQ4_XS Q2_K_L IQ3_M IQ2_XS \
       CD-Q6_K CD-Q5_K_M CD-Q4_K_M CD-Q3_K_L CD-Q2_K)

echo "[build $(ts)] START imatrix-everywhere cohort build (${#TIERS[@]} tiers), floor=${FLOOR_GB}G, free=$(freeGB)G"
[ -f "$GGUFDIR/imatrix.dat" ] || { echo "FATAL: no pre-staged imatrix.dat"; exit 2; }
[ -e "$MODEL/config.json" ]   || { echo "FATAL: model symlink broken"; exit 2; }

for T in "${TIERS[@]}"; do
  [ -f "$DONE/$T.done" ] && { echo "[build $(ts)] skip $T (.done)"; continue; }
  fr=$(freeGB)
  if [ "${fr:-0}" -lt "$FLOOR_GB" ]; then
    echo "[build $(ts)] DISK_FLOOR_STOP before $T (free=${fr}G < ${FLOOR_GB}G); remaining wait for publish+delete"
    echo "BUILD_PAUSED_DISK"; exit 0
  fi
  echo "[build $(ts)] >>> building $T (free=${fr}G)"
  if "$PY" "$WRAP" --model "$MODEL" --output-dir "$GGUFDIR" --only "$T" \
        --base-model-id "$BASEID" --no-upload --keep-local --ngl 0 --sanity-check \
        > "$WORK/build_$T.log" 2>&1; then
    f="$GGUFDIR/gemma-4-A4B-98e-v7-coder-it-$T.gguf"
    if [ -f "$f" ]; then
      touch "$DONE/$T.done"
      san=$(grep -ciE "sanity.*(pass|ok)|all .* correct|PASS" "$WORK/build_$T.log")
      echo "[build $(ts)] TIER_DONE $T size=$(du -h "$f"|cut -f1) sanity_hits=$san free=$(freeGB)G"
    else
      echo "[build $(ts)] TIER_FAIL $T (rc=0 but no gguf)"; tail -6 "$WORK/build_$T.log"
    fi
  else
    echo "[build $(ts)] TIER_FAIL $T (rc!=0)"; tail -10 "$WORK/build_$T.log"
  fi
done
echo "[build $(ts)] ALL_BUILD_DONE"
