#!/usr/bin/env bash
# Q6_K WITH imatrix for every REAM arm -- matching how the two anchors were actually built.
#
# Why this replaces the no-imat chain: reading the GGUF KV of the two pre-existing anchors
# (256e base, published 184e) shows both were quantized WITH an imatrix from
# calibration_datav5.txt (510 entries / 128 chunks). The base column is the anchor the whole
# table hangs on, so putting the arms on a different quant recipe than the base would put a
# recipe difference underneath every arm-vs-base delta. Our own crossover note says imat costs
# ~0.6pp at Q6 -- but a uniform offset cancels in an arm-vs-arm comparison, whereas a
# base-vs-arms recipe MISMATCH does not. Uniformity wins.
#
# --cal-data is left at the bundled default (scripts/calibration_datav5.txt) = the anchors' corpus.
# imatrix.dat lands in each arm's output dir and is KEPT (mandatory archival: a quant whose
# imatrix is lost cannot be reproduced or audited).
#
# The gate is the ARTIFACT: after each tier we read quantize.imatrix.* out of the GGUF KV.
# Passing --force-imatrix proves intent, not outcome; only the metadata proves outcome.
set -uo pipefail

WORK=/mnt/sdc/ream-work
GG=$WORK/gguf
PY=/srv/ml/envs/envs/omnimergekit/bin/python
QG=/srv/ml/repos/omnimergekit/scripts/quantize_gguf.py
LOG=$WORK/quant_arms_imat.log
THREADS=$(( $(nproc) / 2 )); [ "$THREADS" -ge 4 ] || THREADS=4

# GPU1 is the ONLY device this work may touch; GPU0 belongs to someone else. Polling GPU1
# for free VRAM (below) is not reserving it: unset CUDA_VISIBLE_DEVICES lets quantize_gguf
# sum every physical GPU and lets llama-imatrix offload across both. Pin, do not poll.
export CUDA_VISIBLE_DEVICES=1

exec >>"$LOG" 2>&1
echo "=== imat quant chain start $(date -u +%F' '%T) threads=$THREADS ==="
mkdir -p "$GG"

# imatrix runs on the GPU, so nothing may start until the arm-F build has released GPU1.
waited=0
while ! grep -q ">>> ARMF_BUILT_OK" "$WORK/build_armF.log" 2>/dev/null; do
  grep -q ">>> ARMF_BUILD_FAIL" "$WORK/build_armF.log" 2>/dev/null && { echo "armF build FAILED; continuing without it"; break; }
  sleep 60; waited=$((waited+60))
  [ $((waited % 600)) -eq 0 ] && echo "[$(date -u +%H:%M:%S)] waiting on armF build (${waited}s)"
  [ $waited -ge 7200 ] && { echo "ABORT: no armF sentinel after 2h"; exit 3; }
done
for _ in $(seq 1 30); do
  free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free" -ge 60000 ] && break
  echo "GPU1 ${free}MiB free, waiting for the builder to release it"; sleep 60
done
echo "GPU1 ready after ${waited}s"

ARMS="armD_ourssal_nomerge armC armB armE armF_rnorm_nomerge"

built=0; failed=0; skipped=0
for arm in $ARMS; do
  src=$WORK/$arm
  out=$GG/${arm}_imat
  mkdir -p "$out"

  q=$(ls "$out"/*-Q6_K.gguf 2>/dev/null | head -1)
  if [ -n "$q" ] && [ -s "$q" ]; then
    echo "==== SKIP $arm (Q6_K present)"; skipped=$((skipped+1)); continue
  fi
  [ -s "$src/config.json" ] || { echo "==== SKIP $arm (no config.json)"; failed=$((failed+1)); continue; }

  avail=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
  if [ "$avail" -lt 85 ]; then
    echo "==== ABORT $arm: /mnt/sdc ${avail}G free, need ~80G transient"; failed=$((failed+1)); break
  fi

  echo ">>>> $(date -u +%H:%M:%S) START $arm (avail ${avail}G)"
  # No --keep-local: quantize_gguf drops its own F16 intermediate afterwards.
  OMK_NO_README=1 nice -n 10 "$PY" "$QG" --model "$src" --only Q6_K \
      --output-dir "$out" --base-precision f16 --no-upload --force-imatrix \
      --base-model-id "ManniX-ITA/Qwen3.6-27B-A3B-Coder" --threads "$THREADS"
  echo "<<<< $(date -u +%H:%M:%S) END $arm rc=$?"

  q=$(ls "$out"/*-Q6_K.gguf 2>/dev/null | head -1)
  if [ -z "$q" ] || [ ! -s "$q" ] || [ "$(head -c4 "$q")" != "GGUF" ]; then
    echo "     FAIL $arm: no valid Q6_K produced"; failed=$((failed+1)); continue
  fi
  # ARTIFACT gate: the imatrix must be recorded in the GGUF, not merely requested.
  if "$PY" - "$q" <<'PYEOF'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
keys = {k: f for k, f in r.fields.items() if k.startswith("quantize.imatrix")}
if not keys:
    print("     IMATRIX MISSING from GGUF KV"); sys.exit(1)
for k, f in sorted(keys.items()):
    try: print(f"     {k} = {f.contents()}")
    except Exception: print(f"     {k} = ?")
sys.exit(0)
PYEOF
  then
    im=$out/imatrix.dat
    [ -s "$im" ] && echo "     imatrix.dat preserved ($(stat -c %s "$im" | numfmt --to=iec))" \
                 || echo "     WARNING: imatrix.dat NOT in $out -- archival rule violated"
    echo "     OK $arm -> $q ($(stat -c %s "$q" | numfmt --to=iec))"
    built=$((built+1))
  else
    echo "     FAIL $arm: Q6_K built WITHOUT an imatrix -- wrong basis, not counting it"
    failed=$((failed+1))
  fi
done

echo "=== imat quant chain done $(date -u +%F' '%T) built=$built skipped=$skipped failed=$failed ==="
echo ">>> QUANT_ARMS_DONE built=$built failed=$failed"
