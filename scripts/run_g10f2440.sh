#!/usr/bin/env bash
# run_g10f2440.sh — build g10f2440 (wt1.0, floor24-40) + gated eval on GPU1 after g15f2440.
set -uo pipefail
BM=/srv/ml
exec > >(tee -a "$BM/logs/g10f2440_orch.log") 2>&1
B=$BM/scripts/build_v7coder_gsweep.sh
CC=$BM/scripts/canon_chain.sh
TPL=(gpqa_diamond_full lcb_medium_55_v4 ifeval_100 humanevalplus_full multipl_e_100)
O(){ echo "[g10f2440-orch $(date -u +%H:%M:%S)] $*"; }
O "=== BUILD g10f2440 (wt1.0 floor24-40) ==="
bash "$B" g10f2440 1 24 40 || { O "BUILD FAIL g10f2440"; exit 1; }
O "=== BUILD done; arming eval (GPU1, gated on g15f2440 done) ==="
GG=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g10f2440-it-GGUF/gemma-4-A4B-98e-v7-coder-g10f2440-it-Q6_K.gguf
TK=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g10f2440-it
RS=$BM/eval_results_v7coder_g10f2440
setsid bash "$CC" 1 8202 "$GG" "$TK" v7coder-g10f2440-q6k "$RS" \
  "$BM/logs/chain_g15f2440.log" "CANON_CHAIN_v7coder-g15f2440-q6k_G1_DONE" \
  "${TPL[@]}" > "$BM/logs/chain_g10f2440.log" 2>&1 &
O "=== eval armed (chain detached); orchestrator exiting ==="
