#!/usr/bin/env bash
# Phase-1 Q6_K apples-to-apples gap-fill on bs2 (Blackwell), greedy llama.cpp.
# Fills the missing {model x bench} cells for the v7 cohort card comparison:
#   - v7-coder (g15f2440): 6 missing benches on GPU1.
#   - 128e: ALL 11 benches, split GPU0(5)+GPU1(6), each chain self-gated.
# v7-coderx (fs2440) and v6-coder (bs2built) are already 11/11 on bs2.
#
# canon_chain.sh self-gates on (GATE_LOG, GATE_MARK); launch all up front.
# Launch:  setsid nohup bash eval_gapfill_q6k.sh >LOG 2>&1 </dev/null &
set -uo pipefail
GG=/mnt/sdc/ml/eval_gguf
CH=/srv/ml/scripts/canon_chain.sh
LOGS=/srv/ml/logs
PUBLOG=$LOGS/publish_v7coder.log
V7Q6=$GG/v7coder-g15f2440-Q6_K.gguf
V7TOK=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it
E128Q6=$GG/128e-Q6_K.gguf
E128TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
RES_V7=/srv/ml/eval_results_v7coder_g15f2440
RES_128=/srv/ml/eval_results_128e_bs2
V7LOG=$LOGS/eval_gap_v7coder.log
E128G0LOG=$LOGS/eval_gap_128e_g0.log
E128G1LOG=$LOGS/eval_gap_128e_g1.log
mkdir -p "$RES_128"
L(){ echo "[evalgap $(date -u +%H:%M:%S)] $*"; }

wait_stable(){  # $1 file: wait until exists and size stable across 30s and > 1GB
  local f="$1" s1 s2
  L "waiting for $f"
  while :; do
    if [ -f "$f" ]; then
      s1=$(stat -c %s "$f" 2>/dev/null || echo 0); sleep 30; s2=$(stat -c %s "$f" 2>/dev/null || echo 0)
      [ "$s1" = "$s2" ] && [ "${s1:-0}" -gt 1000000000 ] && { L "ready: $f ($s2 bytes)"; return 0; }
    else sleep 20; fi
  done
}
wait_done(){ local log="$1" mark="$2"; while ! grep -q "$mark" "$log" 2>/dev/null; do sleep 30; done; }

L "###### Phase-1 Q6_K gap-fill START ######"

# --- v7-coder gap-fill: GPU1, 6 benches, no gate ---
wait_stable "$V7Q6"
L "launch v7-coder gap-fill GPU1 (6 benches)"
setsid bash "$CH" 1 8211 "$V7Q6" "$V7TOK" v7coder-g15f2440-q6k "$RES_V7" "" "" \
  gsm8k_100 math500_100 aime_30 arc_challenge_full humaneval_full lcb_medium_100_v4 \
  >"$V7LOG" 2>&1 </dev/null &

# --- 128e full 11-bench, split across both GPUs, each self-gated ---
wait_stable "$E128Q6"
L "launch 128e GPU0 (5 benches, gated on publish 'DONE v7-coderx')"
setsid bash "$CH" 0 8210 "$E128Q6" "$E128TOK" 128e-q6k "$RES_128" "$PUBLOG" "DONE v7-coderx" \
  gpqa_diamond_full aime_30 math500_100 ifeval_100 gsm8k_100 \
  >"$E128G0LOG" 2>&1 </dev/null &
L "launch 128e GPU1 (6 benches, gated on v7-coder gap-fill done)"
setsid bash "$CH" 1 8212 "$E128Q6" "$E128TOK" 128e-q6k "$RES_128" "$V7LOG" "CANON_CHAIN_v7coder-g15f2440-q6k_G1_DONE" \
  arc_challenge_full humaneval_full humanevalplus_full lcb_medium_55 lcb_medium_100 multipl_e_100 \
  >"$E128G1LOG" 2>&1 </dev/null &

# --- wait for all three chains ---
wait_done "$V7LOG"    "CANON_CHAIN_v7coder-g15f2440-q6k_G1_DONE"; L "v7-coder gap-fill DONE"
wait_done "$E128G0LOG" "CANON_CHAIN_128e-q6k_G0_DONE";            L "128e GPU0 DONE"
wait_done "$E128G1LOG" "CANON_CHAIN_128e-q6k_G1_DONE";            L "128e GPU1 DONE"
L "###### PHASE1_GAPFILL_ALL_DONE ######"
