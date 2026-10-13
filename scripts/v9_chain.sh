#!/usr/bin/env bash
# v9_chain.sh — wait for the Q6_K, then run the LCB eval (GPU0) and the Q4_K_M (CPU) in
# parallel. Q4_K_M is --ngl 0 so it cannot contend with the eval for GPU memory.
set -uo pipefail
GG=/mnt/sdc/ml/v9build/v9-agentic-GGUF
ts(){ date -u '+%H:%M:%SZ'; }

echo "[chain $(ts)] waiting for Q6_K + V9BUILD_DONE ..."
for i in $(seq 1 360); do   # up to 3h
  if grep -q "V9BUILD_DONE" /mnt/sdc/harness/v9build.log 2>/dev/null; then break; fi
  if grep -qE "^FATAL" /mnt/sdc/harness/v9build.log 2>/dev/null; then
    echo "[chain $(ts)] BUILD FAILED — aborting chain"; grep -E "^FATAL" /mnt/sdc/harness/v9build.log; exit 1
  fi
  sleep 30
done
Q6=$(ls "$GG"/*Q6_K*.gguf 2>/dev/null | head -1)
[ -n "$Q6" ] || { echo "[chain $(ts)] FATAL no Q6_K after wait"; exit 1; }
echo "[chain $(ts)] Q6_K ready: $Q6 ($(stat -c%s "$Q6") bytes)"

echo "[chain $(ts)] launching LCB eval (GPU0) and Q4_K_M (CPU) in parallel"
nohup bash /mnt/sdc/harness/run_lcb_v9.sh "$Q6" > /mnt/sdc/harness/v9lcb.log 2>&1 &
E=$!
nohup bash /mnt/sdc/harness/build_v9_q4km.sh > /mnt/sdc/harness/v9q4km.log 2>&1 &
K=$!
wait $K; echo "[chain $(ts)] Q4KM_RC=$?"
wait $E; echo "[chain $(ts)] LCB_RC=$?"
echo "###### V9_CHAIN_DONE $(ts) ######"
