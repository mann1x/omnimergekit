#!/usr/bin/env bash
# pod_v10_chain.sh — ON THE POD: order the v10 quick builds around the running IFEval sweep and the disk.
#   base fetch -> bf16 hub_k12 (CPU, during the sweep) -> wait for both sweep queues + the llama.cpp b11707 build
#   -> drop the sweep's re-downloadable GGUF copies (Q4_K_M, CD-Q3_K_L, IQ2_S; HF has them — the copies of the
#   three tiers deleted from HF are KEPT) -> bf16 hub_k9 -> remove the base -> quant hub_k12 -> quant hub_k9.
# Each stage needs a free-disk floor; any failure stops the chain with V10_FAIL / CHAIN_FAIL.
set -uo pipefail
W=/workspace; S=$W/omnimergekit/scripts/pod_v10_quick_build.sh; LOG=$W/v10/logs
say(){ echo "[chain $(date -u +%FT%TZ)] $*"; }
need(){ local f; f=$(df --output=avail -BG $W | tail -n1 | tr -dc 0-9); [ "$f" -ge "$1" ] || { say "CHAIN_FAIL disk ${f}G < $1G before $2"; exit 1; }; say "disk ${f}G free (need $1G) before $2"; }
run(){ bash $S "$@" || { say "CHAIN_FAIL stage $*"; exit 1; }; }
mkdir -p $LOG
need 75 base;            run base
need 62 "bf16 hub_k12";  run bf16 hub_k12
say "waiting for both IFEval sweep queues and the llama.cpp b11707 build"
for _ in $(seq 1 1440); do
  d=$(cat $W/v9qmpe/logs/sweep_gpu0.log $W/v9qmpe/logs/sweep_gpu1.log 2>/dev/null | grep -c SWEEP_DONE)
  b=$(grep -c "LCBUILD_RC=" $W/v9qmpe/logs/lclatest_build.log 2>/dev/null)
  [ "$d" -ge 2 ] && [ "$b" -ge 1 ] && break
  sleep 30
done
[ "$d" -ge 2 ] || { say "CHAIN_FAIL sweep not done after 12 h"; exit 1; }
grep -q "LCBUILD_RC=0" $W/v9qmpe/logs/lclatest_build.log || { say "CHAIN_FAIL llama.cpp b11707 build failed"; exit 1; }
for t in Q4_K_M CD-Q3_K_L IQ2_S; do rm -f $W/v9qmpe/gguf/gemma-4-20B-v9-agentic-$t.gguf $W/v9qmpe/gguf/gemma-4-20B-v9-agentic-$t.gguf.ok; done
say "sweep done; removed the re-downloadable copies of Q4_K_M, CD-Q3_K_L, IQ2_S"
need 50 "bf16 hub_k9";   run bf16 hub_k9
run rmbase
need 60 "quant hub_k12"; run quant hub_k12
need 60 "quant hub_k9";  run quant hub_k9
say "V10_CHAIN_DONE"
