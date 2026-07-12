#!/usr/bin/env bash
# run_gsweep.sh — build + eval 3 fs2440+gpqa sweep variants on 2 GPUs.
#   A g10f2035 (wt1.0, floor20-35)  GPU0 first
#   C g15f2440 (wt1.5, floor24-40)  GPU1   <- direct fs2440-family weight comparison
#   B g15f2035 (wt1.5, floor20-35)  GPU0 gated-after-A  <- floor isolator vs C
set -uo pipefail
BM=/srv/ml
exec > >(tee -a "$BM/logs/gsweep_orch.log") 2>&1
B=$BM/scripts/build_v7coder_gsweep.sh
CC=$BM/scripts/canon_chain.sh
TPL=(gpqa_diamond_full lcb_medium_55_v4 ifeval_100 humanevalplus_full multipl_e_100)
O(){ echo "[orch $(date -u +%H:%M:%S)] $*"; }
gg(){ echo /mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-$1-it-GGUF/gemma-4-A4B-98e-v7-coder-$1-it-Q6_K.gguf; }
tk(){ echo /mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-$1-it; }
rs(){ echo $BM/eval_results_v7coder_$1; }

O "=== PHASE 1: builds (sequential, CPU) ==="
bash "$B" g10f2035 1   20 35 || { O "BUILD FAIL g10f2035"; exit 1; }
bash "$B" g15f2440 1.5 24 40 || { O "BUILD FAIL g15f2440"; exit 1; }
bash "$B" g15f2035 1.5 20 35 || { O "BUILD FAIL g15f2035"; exit 1; }
O "=== PHASE 1 done; all 3 Q6 built ==="

LA=$BM/logs/chain_g10f2035.log
LB=$BM/logs/chain_g15f2035.log
LC=$BM/logs/chain_g15f2440.log
O "=== PHASE 2: evals (A->GPU0, C->GPU1, B->GPU0 gated-on-A) ==="
setsid bash "$CC" 0 8201 "$(gg g10f2035)" "$(tk g10f2035)" v7coder-g10f2035-q6k "$(rs g10f2035)" "" "" "${TPL[@]}" > "$LA" 2>&1 &
setsid bash "$CC" 1 8202 "$(gg g15f2440)" "$(tk g15f2440)" v7coder-g15f2440-q6k "$(rs g15f2440)" "" "" "${TPL[@]}" > "$LC" 2>&1 &
setsid bash "$CC" 0 8201 "$(gg g15f2035)" "$(tk g15f2035)" v7coder-g15f2035-q6k "$(rs g15f2035)" "$LA" "CANON_CHAIN_v7coder-g10f2035-q6k_G0_DONE" "${TPL[@]}" > "$LB" 2>&1 &
wait
O "=== ALL_DONE: gsweep complete ==="
