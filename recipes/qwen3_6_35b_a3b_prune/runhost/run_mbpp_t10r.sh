#!/usr/bin/env bash
# MBPP-500 SAME-CONFIG REPEAT (t8 -> t8r). The missing band control.
#
# REPEAT OF THE t10 ARM. I hold two t8 draws (0.784/0.790, band = net +3 / churn 21) but only
# ONE t10 draw. A release decision must not rest on a single draw of the losing arm -- that is
# the exact trap the LCB cohort fell into. Same flags as armJ_g_t10, override INCLUDED.
# ORIGINAL NOTE: mbpp t8=0.784 vs t10=0.732 is -5.2pp, 41 lost / 15 gained, McNemar p=0.000686 -- the
# first result in this routing study that looks systematic. But greedy is NOT
# bit-reproducible under batching, and the only measured band for this family (+/-0.16pp,
# greedy MPE @p4) comes from a DIFFERENT bench on a DIFFERENT artifact. Without a t8-vs-t8
# MBPP repeat on armJ I cannot separate a routing effect from this bench's engine jitter.
# Identical flags to the t8 cell -- the ONLY thing that differs is that it is a second run.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
export HF_ALLOW_CODE_EVAL=1
OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
WORK=/mnt/sdc/ream-work
RES=/srv/ml/eval_results_routing
LOG=$WORK/run_mbpp_t10r.log
FLOG=/mnt/sdc/ream-work/run_mbpp_t10r.log
ARMJ=$WORK/gguf/armJ_imat/armJ-Q6_K.gguf
CTX=98304
PAR=4
PORT=8099
CELL=armJ_g_t10r
OUT=$RES/qwen_suite/mbpp_full/$CELL
say(){ echo "[mbppband $(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# No sentinel wait: the greedy cohort, gap-fill and band runs have all finished and GPU1 is
# idle. The sed that produced this file pointed the wait at THIS script's own completion
# sentinel, which would deadlock. The real guards below (no live llama-server + the
# bug-617 GPU-settle predicate) are what actually matter.
for i in $(seq 1 30); do
  pgrep -f "llama[-]server" >/dev/null 2>&1 || break   # bug-614: split pattern, no self-match
  say "  a llama-server is still alive; waiting"
  sleep 20
done

# bug-617: a PREDICATE, not a sleep. Teardown util drains asynchronously and omk's
# gpu_planner refuses to launch at util >= 15% even when memory already reads 0 MiB.
settled=0
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1)
  if [ "${used:-99999}" -lt 2000 ] && [ "${util:-100}" -lt 10 ]; then
    say "GPU1 settled (mem=${used}MiB util=${util}%) after $((i*5))s"
    settled=1
    break
  fi
  sleep 5
done
[ "$settled" = "1" ] || { say "REFUSE: GPU1 never settled in 300s"; exit 1; }

if [ -f "$OUT/summary.json" ]; then
  say "SKIP mbpp_full/$CELL (already done)"
  say "MBPP_T10R_DONE"
  exit 0
fi
[ -d "$OUT" ] && mv "$OUT" "${OUT}_PARTIAL_$(date -u +%Y%m%dT%H%M%SZ)"

TOK=/srv/ml/models/Qwen3.6-35B-A3B
[ -d "$WORK/armJ" ] && TOK=$WORK/armJ

say "===== mbpp_full/$CELL topk=10 REPEAT (identical flags to armJ_g_t10, no --sampler)"
"$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template mbpp_full --quant q6_k \
    --model "$ARMJ" --tokenizer "$TOK" --served-name "$CELL" --port "$PORT" \
    --results-dir "$RES/qwen_suite" --parallel "$PAR" --gpus 1 \
    --metadata backend_args.llama_ctx=$CTX \
    --metadata backend_args.llama_extra='["--override-kv","qwen35moe.expert_used_count=int:10"]' 2>&1 | tail -5

if [ ! -f "$OUT/server.log" ]; then
  say "FAIL mbpp_full/$CELL: no server.log -- died before serving"
  exit 1
fi
printf '{"intended_topk":10,"llama_extra":["--override-kv","qwen35moe.expert_used_count=int:10"],"sampler":"template_default (greedy)","parallel":%s,"llama_ctx":%s,"note":"SAME-CONFIG REPEAT of armJ_g_t10 -- confirms the t10 arm is not a single unlucky draw"}\n' \
  "$PAR" "$CTX" > "$OUT/routing.json"
if ls "$OUT" | grep -q "server.gpu0"; then say "ALERT mbpp_full/$CELL: gpu0 server log present -- pin leaked!"; fi

if [ -f "$OUT/summary.json" ]; then
  SCORE=$("$OMKPY" "$WORK/read_score.py" "$OUT/summary.json")
  say "SCORE mbpp_full/$CELL = $SCORE"
else
  say "FAIL mbpp_full/$CELL: no summary.json"
fi
say "MBPP_T10R_DONE"
