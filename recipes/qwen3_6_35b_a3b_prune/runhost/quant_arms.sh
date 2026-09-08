#!/usr/bin/env bash
# Q6_K (no-imatrix) for every REAM arm, so the 6-arm table sits on ONE quant basis.
#
# Why no-imatrix: an imatrix is computed per model, so 6 arms would mean 6 different
# imatrices and the quant recipe would vary with the thing under test. Q6 is above our
# measured imatrix crossover (noimat ~= imat at 6 bits), so dropping it removes a
# confound instead of adding one. The two pre-existing imat Q6_K files (256e base,
# published 184e) stay untouched and are run as separate BRIDGE columns -- armD-noimat
# vs published-imat is bit-identical weights under two quant recipes, which measures the
# imat effect on this exact model rather than assuming it away.
#
# CPU-only (nice'd, half the cores) so it can run alongside the arm-F build on GPU1.
set -uo pipefail

WORK=/mnt/sdc/ream-work
GG=$WORK/gguf
PY=/srv/ml/envs/envs/omnimergekit/bin/python
QG=/srv/ml/repos/omnimergekit/scripts/quantize_gguf.py
LOG=$WORK/quant_arms.log
THREADS=$(( $(nproc) / 2 )); [ "$THREADS" -ge 4 ] || THREADS=4

exec >>"$LOG" 2>&1
echo "=== quant chain start $(date -u +%F' '%T) threads=$THREADS ==="
mkdir -p "$GG"

# armF is built by build_arm_f.sh on the GPU; it is last so the CPU work overlaps that build.
ARMS="armC armB armE armD_ourssal_nomerge armF_rnorm_nomerge"

built=0; failed=0; skipped=0
for arm in $ARMS; do
  src=$WORK/$arm
  out=$GG/$arm
  mkdir -p "$out"

  if ls "$out"/*-Q6_K.gguf >/dev/null 2>&1; then
    echo "==== SKIP $arm (Q6_K present)"; skipped=$((skipped+1)); continue
  fi

  # armF may still be building. Wait for its sentinel rather than converting a half-written dir.
  if [ "$arm" = "armF_rnorm_nomerge" ]; then
    waited=0
    while [ ! -d "$src" ] || ! grep -q ">>> ARMF_BUILT_OK" "$WORK/build_armF.log" 2>/dev/null; do
      if grep -q ">>> ARMF_BUILD_FAIL" "$WORK/build_armF.log" 2>/dev/null; then
        echo "==== SKIP $arm -- its build FAILED, refusing to quantize a broken arm"
        failed=$((failed+1)); break
      fi
      sleep 60; waited=$((waited+60))
      [ $waited -ge 7200 ] && { echo "==== ABORT $arm: no build sentinel after 2h"; failed=$((failed+1)); break; }
    done
    grep -q ">>> ARMF_BUILT_OK" "$WORK/build_armF.log" 2>/dev/null || continue
    echo "armF build sentinel seen after ${waited}s"
  fi

  [ -s "$src/config.json" ] || { echo "==== SKIP $arm (no config.json at $src)"; failed=$((failed+1)); continue; }

  # An F16 intermediate is ~54G on top of the ~22G tier. Refuse rather than fill the disk.
  avail=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
  if [ "$avail" -lt 90 ]; then
    echo "==== ABORT $arm: /mnt/sdc only ${avail}G free, need ~80G transient"; failed=$((failed+1)); break
  fi

  echo ">>>> $(date -u +%H:%M:%S) START $arm (avail ${avail}G)"
  OMK_NO_README=1 nice -n 19 "$PY" "$QG" --model "$src" --only Q6_K \
      --output-dir "$out" --base-precision f16 --no-upload --no-imatrix \
      --base-model-id "ManniX-ITA/Qwen3.6-27B-A3B-Coder" --threads "$THREADS"
  rc=$?
  echo "<<<< $(date -u +%H:%M:%S) END $arm rc=$rc"

  # Gate on the artifact, not the exit code.
  q=$(ls "$out"/*-Q6_K.gguf 2>/dev/null | head -1)
  if [ -s "$q" ] && [ "$(head -c4 "$q")" = "GGUF" ]; then
    echo "     OK $arm -> $q ($(stat -c %s "$q" | numfmt --to=iec))"
    built=$((built+1))
    # Drop the F16 intermediate; 5 of them would not fit and they are reproducible.
    find "$out" -maxdepth 1 -type f -iname '*f16*.gguf' -print -delete
  else
    echo "     FAIL $arm: no valid Q6_K produced"; failed=$((failed+1))
  fi
done

echo "=== quant chain done $(date -u +%F' '%T) built=$built skipped=$skipped failed=$failed ==="
echo ">>> QUANT_ARMS_DONE built=$built failed=$failed"
