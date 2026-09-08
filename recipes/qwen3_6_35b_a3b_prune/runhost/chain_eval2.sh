#!/usr/bin/env bash
# Queue behind the imat quant chain: (1) rebuild the noimat control, (2) run the R2 eval.
#
# GATE = ARTIFACTS, NOT THE LOG SENTINEL. The first version of this script grepped
# ">>> QUANT_ARMS_DONE" and fired instantly, because the log still carried that line from the
# EARLIER aborted run (built=2 failed=1). A sentinel that is not unique per run is not a gate.
# So this waits until all five arm Q6_K files exist with their .sha256 sidecar written --
# state that can only be true of the current run, since armB/E/F had no Q6_K before it.
#
# Why step 1 exists: gguf/armC_noimat/armC-Q6_K.gguf was built 05:36, BEFORE the MTP graft.
# Verified on disk: block_count=41 but 0 blk.40.* tensors -- the exact defect that made every
# arm unservable ('missing tensor blk.40.attn_norm.weight'). eval_ream_arms.sh lists it as the
# reamC_noimat_ctrl column, so leaving it would spend a slot producing another dead column.
# It is rebuilt from the SAME grafted armC weights the imat column uses, so the only difference
# between reamC_ourssal_merge and reamC_noimat_ctrl stays the imatrix -- the point of the
# control (T118 / #411).
set -uo pipefail
WORK=/mnt/sdc/ream-work
GG=$WORK/gguf
PY=/srv/ml/envs/envs/omnimergekit/bin/python
QG=/srv/ml/repos/omnimergekit/scripts/quantize_gguf.py
LOG=$WORK/chain_eval2.log
THREADS=$(( $(nproc) / 2 )); [ "$THREADS" -ge 4 ] || THREADS=4
exec >>"$LOG" 2>&1
echo "=== chain_eval2 start $(date -u +%F' '%T) ==="

Q6=( "$GG/armD_ourssal_nomerge_imat/armD_ourssal_nomerge-Q6_K.gguf"
     "$GG/armC_imat/armC-Q6_K.gguf"
     "$GG/armB_imat/armB-Q6_K.gguf"
     "$GG/armE_imat/armE-Q6_K.gguf"
     "$GG/armF_rnorm_nomerge_imat/armF_rnorm_nomerge-Q6_K.gguf" )

ready() { local n=0; for q in "${Q6[@]}"; do [ -s "$q" ] && [ -s "${q}.sha256" ] && n=$((n+1)); done; echo "$n"; }

waited=0
while [ "$(ready)" -lt 5 ]; do
  sleep 60; waited=$((waited+60))
  [ $((waited % 600)) -eq 0 ] && echo "[$(date -u +%H:%M:%S)] $(ready)/5 arm Q6_K ready (${waited}s)"
  if [ $waited -ge 21600 ]; then
    echo "ABORT: only $(ready)/5 arm Q6_K after 6h"; exit 3
  fi
done
echo "all 5 arm Q6_K present after ${waited}s"

blk40() { "$PY" - "$1" <<'PYEOF'
import sys
from gguf import GGUFReader
print(sum(1 for t in GGUFReader(sys.argv[1]).tensors if t.name.startswith("blk.40.")))
PYEOF
}

echo "--- MTP presence per arm (want 20 blk.40 tensors each)"
for q in "${Q6[@]}"; do echo "    $(blk40 "$q")  $(basename "$(dirname "$q")")"; done

q=$GG/armC_noimat/armC-Q6_K.gguf
n=0; [ -s "$q" ] && n=$(blk40 "$q")
if [ "$n" -eq 20 ]; then
  echo "noimat ctrl already carries 20 blk.40 tensors -- no rebuild"
else
  echo ">>>> $(date -u +%H:%M:%S) REBUILD armC_noimat (had ${n} blk.40 tensors, want 20)"
  rm -f "$GG"/armC_noimat/armC-Q6_K.gguf "$GG"/armC_noimat/armC-Q6_K.gguf.sha256 \
        "$GG"/armC_noimat/armC-F16.gguf
  OMK_NO_README=1 nice -n 10 "$PY" "$QG" --model "$WORK/armC" --only Q6_K \
      --output-dir "$GG/armC_noimat" --base-precision f16 --no-upload --no-imatrix \
      --base-model-id "ManniX-ITA/Qwen3.6-27B-A3B-Coder" --threads "$THREADS"
  echo "<<<< $(date -u +%H:%M:%S) rebuild rc=$?"
  n=0; [ -s "$q" ] && n=$(blk40 "$q")
  if [ "$n" -ne 20 ]; then
    echo "FAIL: noimat ctrl still has ${n} blk.40 tensors -- eval runs WITHOUT a valid ctrl column"
  else
    echo "OK noimat ctrl: 20 blk.40 tensors, $(du -h "$q" | cut -f1)"
    f=$(ls "$GG"/armC_noimat/*-F16.gguf 2>/dev/null | head -1)
    [ -n "$f" ] && [ -s "${q}.sha256" ] && { rm -f "$f"; echo "reaped $(basename "$f")"; }
  fi
fi

for _ in $(seq 1 30); do
  free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free" -ge 60000 ] && break
  echo "GPU1 only ${free}MiB free, waiting"; sleep 60
done

bash "$WORK/eval_ream_arms.sh"
echo "=== chain_eval2 done $(date -u +%F' '%T) rc=$? ==="
