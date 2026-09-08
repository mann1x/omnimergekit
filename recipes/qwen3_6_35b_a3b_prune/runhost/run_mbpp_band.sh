#!/usr/bin/env bash
# MBPP-500 SAME-CONFIG REPEAT (t8 -> t8r). The missing band control.
#
# WHY: mbpp t8=0.784 vs t10=0.732 is -5.2pp, 41 lost / 15 gained, McNemar p=0.000686 -- the
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
LOG=$WORK/run_mbpp_band.log
FLOG=$WORK/run_routing_greedy_fix.log
ARMJ=$WORK/gguf/armJ_imat/armJ-Q6_K.gguf
CTX=98304
PAR=4
PORT=8099
CELL=armJ_g_t8r
OUT=$RES/qwen_suite/mbpp_full/$CELL
say(){ echo "[mbppband $(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

say "waiting for ROUTING_GFIX_DONE"
for i in $(seq 1 720); do
  grep -q "ROUTING_GFIX_DONE" "$FLOG" 2>/dev/null && { say "gfix finished"; break; }
  sleep 60
done
if ! grep -q "ROUTING_GFIX_DONE" "$FLOG" 2>/dev/null; then
  say "REFUSE: gfix never signalled -- not racing it for the GPU"
  exit 1
fi
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
  say "MBPP_BAND_DONE"
  exit 0
fi
[ -d "$OUT" ] && mv "$OUT" "${OUT}_PARTIAL_$(date -u +%Y%m%dT%H%M%SZ)"

TOK=/srv/ml/models/Qwen3.6-35B-A3B
[ -d "$WORK/armJ" ] && TOK=$WORK/armJ

say "===== mbpp_full/$CELL topk=8 REPEAT (identical flags to armJ_g_t8, no --sampler)"
"$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template mbpp_full --quant q6_k \
    --model "$ARMJ" --tokenizer "$TOK" --served-name "$CELL" --port "$PORT" \
    --results-dir "$RES/qwen_suite" --parallel "$PAR" --gpus 1 \
    --metadata backend_args.llama_ctx=$CTX 2>&1 | tail -5

if [ ! -f "$OUT/server.log" ]; then
  say "FAIL mbpp_full/$CELL: no server.log -- died before serving"
  exit 1
fi
printf '{"intended_topk":8,"llama_extra":[],"sampler":"template_default (greedy)","parallel":%s,"llama_ctx":%s,"note":"SAME-CONFIG REPEAT of armJ_g_t8 -- this pair IS the MBPP engine-jitter band on armJ"}\n' \
  "$PAR" "$CTX" > "$OUT/routing.json"
if ls "$OUT" | grep -q "server.gpu0"; then say "ALERT mbpp_full/$CELL: gpu0 server log present -- pin leaked!"; fi

if [ -f "$OUT/summary.json" ]; then
  SCORE=$("$OMKPY" "$WORK/read_score.py" "$OUT/summary.json")
  say "SCORE mbpp_full/$CELL = $SCORE"
else
  say "FAIL mbpp_full/$CELL: no summary.json"
fi
say "MBPP_BAND_DONE"
