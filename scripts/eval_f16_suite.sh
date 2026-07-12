#!/usr/bin/env bash
# Phase-3 F16 (full-precision GGUF) canonical suite on bs2 (Blackwell), greedy llama.cpp.
# Produces the fp16 comparison table for the v7 cohort card:
#   128e / v7-coder (g15f2440) / v7-coderx (fs2440), 11 benches each (9 canonical + LCB-100 + MPE-100).
# Each model's 11 benches are split 5(GPU0)/6(GPU1); the 3 models are PIPELINED per GPU
# (128e -> v7-coder -> v7-coderx), each downstream chain self-gating on the upstream
# chain's CANON_CHAIN_<served>_G<gpu>_DONE marker (same GPU freed). canon_chain.sh boots
# one llama-server per GPU (CUDA_VISIBLE_DEVICES-pinned) and has a server-death watchdog.
#
# LCB template MATCHES each model's published Q6_K run (apples-to-apples F16-vs-Q6_K per model):
#   128e -> lcb_medium_55 / lcb_medium_100   |   prunes -> lcb_medium_55_v4 / lcb_medium_100_v4
#
# Launch:  setsid nohup bash eval_f16_suite.sh >/srv/ml/logs/eval_f16_suite.log 2>&1 </dev/null &
set -uo pipefail
CH=/srv/ml/scripts/canon_chain.sh
LOGS=/srv/ml/logs
RES=/srv/ml/eval_results_f16
mkdir -p "$RES" "$LOGS"

# model rows: served | gguf | tokenizer_dir | lcb55_tpl | lcb100_tpl
E128_GG=/mnt/sdc/ml/eval_gguf/128e-F16.gguf
E128_TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
V7C_GG=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF/gemma-4-A4B-98e-v7-coder-it-F16.gguf
V7C_TOK=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it
V7X_GG=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it-GGUF/gemma-4-A4B-98e-v7-coderx-it-F16.gguf
V7X_TOK=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it

PORT0=8220; PORT1=8221
L(){ echo "[f16suite $(date -u +%H:%M:%S)] $*"; }
wait_done(){ local log="$1" mark="$2"; while ! grep -q "$mark" "$log" 2>/dev/null; do sleep 30; done; }

# preflight: every bin + tokenizer present
for f in "$E128_GG" "$V7C_GG" "$V7X_GG"; do [ -f "$f" ] || { L "FATAL: missing gguf $f"; exit 1; }; done
for d in "$E128_TOK" "$V7C_TOK" "$V7X_TOK"; do [ -f "$d/tokenizer.json" ] || { L "FATAL: missing tokenizer $d"; exit 1; }; done

# GPU0 = 5 benches (incl. gpqa), GPU1 = 6 benches (incl. lcb100 + mpe). $LCB55/$LCB100 per model.
g0(){ echo gpqa_diamond_full math500_100 ifeval_100 humaneval_full "$1"; }            # $1=lcb55 tpl
g1(){ echo aime_30 gsm8k_100 arc_challenge_full humanevalplus_full "$1" multipl_e_100; } # $1=lcb100 tpl

LOG_128_G0=$LOGS/eval_f16_128e_g0.log;  LOG_128_G1=$LOGS/eval_f16_128e_g1.log
LOG_V7C_G0=$LOGS/eval_f16_v7coder_g0.log;  LOG_V7C_G1=$LOGS/eval_f16_v7coder_g1.log
LOG_V7X_G0=$LOGS/eval_f16_v7coderx_g0.log; LOG_V7X_G1=$LOGS/eval_f16_v7coderx_g1.log

L "###### Phase-3 F16 suite START (3 models x 11 benches, dual-GPU pipelined) ######"

# --- 128e: both GPUs, no gate (start now) ---
L "launch 128e-f16 GPU0(5) + GPU1(6)"
setsid bash "$CH" 0 "$PORT0" "$E128_GG" "$E128_TOK" 128e-f16 "$RES" "" "" \
  $(g0 lcb_medium_55)  >"$LOG_128_G0" 2>&1 </dev/null &
setsid bash "$CH" 1 "$PORT1" "$E128_GG" "$E128_TOK" 128e-f16 "$RES" "" "" \
  $(g1 lcb_medium_100) >"$LOG_128_G1" 2>&1 </dev/null &

# --- v7-coder: gated on 128e freeing each GPU ---
L "launch v7coder-f16 GPU0 (gate 128e-G0) + GPU1 (gate 128e-G1)"
setsid bash "$CH" 0 "$PORT0" "$V7C_GG" "$V7C_TOK" v7coder-f16 "$RES" "$LOG_128_G0" CANON_CHAIN_128e-f16_G0_DONE \
  $(g0 lcb_medium_55_v4)  >"$LOG_V7C_G0" 2>&1 </dev/null &
setsid bash "$CH" 1 "$PORT1" "$V7C_GG" "$V7C_TOK" v7coder-f16 "$RES" "$LOG_128_G1" CANON_CHAIN_128e-f16_G1_DONE \
  $(g1 lcb_medium_100_v4) >"$LOG_V7C_G1" 2>&1 </dev/null &

# --- v7-coderx: gated on v7-coder freeing each GPU ---
L "launch v7coderx-f16 GPU0 (gate v7coder-G0) + GPU1 (gate v7coder-G1)"
setsid bash "$CH" 0 "$PORT0" "$V7X_GG" "$V7X_TOK" v7coderx-f16 "$RES" "$LOG_V7C_G0" CANON_CHAIN_v7coder-f16_G0_DONE \
  $(g0 lcb_medium_55_v4)  >"$LOG_V7X_G0" 2>&1 </dev/null &
setsid bash "$CH" 1 "$PORT1" "$V7X_GG" "$V7X_TOK" v7coderx-f16 "$RES" "$LOG_V7C_G1" CANON_CHAIN_v7coder-f16_G1_DONE \
  $(g1 lcb_medium_100_v4) >"$LOG_V7X_G1" 2>&1 </dev/null &

# --- wait for all 6 chains ---
wait_done "$LOG_128_G0" CANON_CHAIN_128e-f16_G0_DONE;     L "128e-f16 GPU0 DONE"
wait_done "$LOG_128_G1" CANON_CHAIN_128e-f16_G1_DONE;     L "128e-f16 GPU1 DONE"
wait_done "$LOG_V7C_G0" CANON_CHAIN_v7coder-f16_G0_DONE;  L "v7coder-f16 GPU0 DONE"
wait_done "$LOG_V7C_G1" CANON_CHAIN_v7coder-f16_G1_DONE;  L "v7coder-f16 GPU1 DONE"
wait_done "$LOG_V7X_G0" CANON_CHAIN_v7coderx-f16_G0_DONE; L "v7coderx-f16 GPU0 DONE"
wait_done "$LOG_V7X_G1" CANON_CHAIN_v7coderx-f16_G1_DONE; L "v7coderx-f16 GPU1 DONE"
L "###### F16_SUITE_ALL_DONE ######"
