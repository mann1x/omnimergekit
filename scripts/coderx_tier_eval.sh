#!/usr/bin/env bash
# coderx_tier_eval.sh <MODE> <GPU> <PORT>   MODE = heplus | loop
#
# Per-tier eval over the LOCAL code4/lcb3 GGUF ladder as it is built (gguf_coderx/),
# claim-based + resumable, dynamic tier discovery (picks up tiers as the CPU build
# finishes them; size-stable check avoids mid-write files). Self-completes when the
# build marker is present AND all present tiers are done.
#   heplus -> HE+164 (humanevalplus_full) + MPE-100 (multipl_e_100), greedy, omk_eval.
#   loop   -> 48-seed b9700 gate at vendor_minp_rep {t0.9,t0.8} (gate_sweep48_minp_p_b9700).
# Two instances (heplus + loop) co-tenant GPU1; run one each. NEVER uploads.
set -uo pipefail
MODE="${1:?usage: coderx_tier_eval.sh <heplus|loop> <GPU> <PORT>}"
GPU="${2:?need GPU}"; PORT="${3:?need PORT}"

GGUF_DIR=/mnt/sdc/ml/cx_std16/gguf_coderx
PREFIX=CX16c4l3-bf16-
TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
PYE=/srv/ml/envs/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
GATE=/srv/ml/scripts/gate_sweep48_minp_p_b9700.sh
BUILD_LOG=/mnt/sdc/ml/coderx_full_build.log
RES=/srv/ml/eval_results_coderx_sweep
LOUT=/mnt/sdc/ml/cx_std16/coderx_loop_sweep/out
WORK=/mnt/sdc/ml/cx_std16/coderx_sweep_work/$MODE
LOG=/mnt/sdc/ml/coderx_tier_eval_${MODE}.log
PARITY="${PARITY_FILE:-}"   # allowlist: only process tiers whose name is a line in this file (v7-coder parity)
export PATH="/srv/ml/envs/envs/omnimergekit/bin:$PATH"
mkdir -p "$RES" "$LOUT" "$WORK/locks" "$WORK/done"
exec >>"$LOG" 2>&1
ts(){ date '+%T %Z'; }
gguf_stable(){ local f=$1 a b; [ -f "$f" ] || return 1; [ "$(head -c4 "$f" 2>/dev/null)" = GGUF ] || return 1
  a=$(stat -c %s "$f"); sleep 3; b=$(stat -c %s "$f"); [ "$a" = "$b" ]; }
build_done(){ grep -q "CODERX_FULLBUILD_DONE" "$BUILD_LOG" 2>/dev/null; }

echo "################ coderx_tier_eval MODE=$MODE GPU$GPU:$PORT $(ts) ################"

eval_heplus(){ # tier gguf
  local T=$1 G=$2 SN="cx16-c4l3-$1" b rc
  for b in humanevalplus_full multipl_e_100; do
    echo "[$(ts)] $T/$b: omk_eval GPU$GPU:$PORT (greedy --parallel 2)"
    CUDA_VISIBLE_DEVICES=$GPU "$PYE" "$OMK" --model "$G" --tokenizer "$TOK" --template "$b" \
      --backend llama --parallel 2 --port "$PORT" --served-name "$SN" --results-dir "$RES"
    rc=$?; echo "[$(ts)] $T/$b exit=$rc"
  done
}
eval_loop(){ # tier gguf
  local T=$1 G=$2
  echo "[$(ts)] $T: b9700 48-seed loop gate (both temps)"
  bash "$GATE" "$G" "$GPU" "$PORT" "$LOUT/$T.json" "coderx-sw-$T"
}

while true; do
  progressed=0; present=0
  for f in "$GGUF_DIR/${PREFIX}"*.gguf; do
    [ -e "$f" ] || continue
    T="${f#*$PREFIX}"; T="${T%.gguf}"
    [ "$T" = "F16" ] && continue
    [ -n "$PARITY" ] && ! grep -qxF "$T" "$PARITY" && continue   # skip off-parity tiers
    present=$((present+1))
    [ -f "$WORK/done/$T.done" ] && continue
    gguf_stable "$f" || continue
    mkdir "$WORK/locks/$T.lock" 2>/dev/null || continue
    if [ "$MODE" = heplus ]; then eval_heplus "$T" "$f"; else eval_loop "$T" "$f"; fi
    touch "$WORK/done/$T.done"; progressed=1
  done
  if build_done; then
    pend=0
    for f in "$GGUF_DIR/${PREFIX}"*.gguf; do T="${f#*$PREFIX}"; T="${T%.gguf}"; [ "$T" = F16 ] && continue
      [ -n "$PARITY" ] && ! grep -qxF "$T" "$PARITY" && continue
      [ -f "$WORK/done/$T.done" ] || pend=$((pend+1)); done
    [ "$pend" = 0 ] && { echo "[$(ts)] all tiers done (build complete)"; break; }
  fi
  [ "$progressed" = 0 ] && sleep 45
done
echo "###### CODERX_TIEREVAL_${MODE}_DONE $(ts) ######"
