#!/usr/bin/env bash
# CoderX (armJ) routing decision -- DRAW B. Repeat of every cell in draw A.
#
# WHY THIS EXISTS: draw A gave ifeval t8 0.93 / t10 0.92 and lcb t8 0.7662 / t10 0.7143.
# The lcb gap (5.2pp) looks like the July "top-10 costs LCB" effect -- but the SAME-CONFIG
# spread on this model is LARGER: banked armJ t8 lcb = 0.8312 (2026-08-20 15:24Z, single
# GPU, no fleet) vs fresh armJ t8 lcb = 0.7662 (2026-08-21 00:22Z, single GPU, no fleet)
# = 6.5pp apart on the same nominal config. With sampling (recommended: temp 0.6/top_p 0.95)
# and --use_cache a no-op under do_sample, every draw is an independent draw. One draw per
# config cannot separate a 5pp effect from a 6.5pp band. Hence draw B.
#
# GATE CHANGE vs draw A (bug-615): the old GATE-W grepped server.log for "--override-kv".
# That log carries NO argv echo and NO model hparams (6540 lines, zero matches for
# "expert"/"override"/n_embd), so the gate could never pass and voided two good cells.
# Replaced by:
#   GATE-M  one-time MECHANISM probe -- greedy, same GGUF, t8 vs t10, require the two
#           outputs to DIFFER. If they are byte-identical the override is inert and the
#           whole matrix is meaningless, so refuse before spending 6h.
#   sidecar per-cell routing.json recording intended_topk + exact llama_extra.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1   # GPU0 is NOT ours. Never rely on --gpus auto.
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH   # lm-eval is a console script (bug-612)
OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
WORK=/mnt/sdc/ream-work
RES=/srv/ml/eval_results_routing
PORT=8099
PROBE_PORT=8098
TOK_BASE=/srv/ml/models/Qwen3.6-35B-A3B
ARMJ=$WORK/gguf/armJ_imat/armJ-Q6_K.gguf
LSRV=/opt/llama.cpp/build/bin/llama-server
LOG=$WORK/run_routing_b.log
OVR="qwen35moe.expert_used_count=int:10"
say(){ echo "[routeB $(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

ARMS=(
 "ifeval_100|armJ_t8_b|8"
 "ifeval_100|armJ_t10_b|10"
 "lcb_v6_77q|armJ_t8_b|8"
 "lcb_v6_77q|armJ_t10_b|10"
)

[ -s "$ARMJ" ] || { say "REFUSE: missing $ARMJ"; exit 1; }
[ -x "$LSRV" ] || { say "REFUSE: missing $LSRV"; exit 1; }
freeg=$(df -BG --output=avail / | tail -1 | tr -dc "0-9")
[ "${freeg:-0}" -ge 210 ] || { say "REFUSE: root fs ${freeg}G free (floor 200G)"; exit 1; }
say "preflight ok: ${#ARMS[@]} cells, disk ${freeg}G, GPU0 at $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) MiB"

# ---------------- GATE-M: is the override mechanism live at all? ----------------
probe(){   # $1 = topk ; echoes the completion text
  local topk="$1" extra=() pid out code i
  [ "$topk" = "10" ] && extra=(--override-kv "$OVR")
  "$LSRV" -m "$ARMJ" --port "$PROBE_PORT" -c 4096 -ngl 99 --jinja --no-warmup \
      "${extra[@]}" > "$WORK/probe_t${topk}.log" 2>&1 &
  pid=$!
  for i in $(seq 1 90); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PROBE_PORT/health" 2>/dev/null)
    [ "$code" = "200" ] && break     # bug-611: curl exits 0 on HTTP 503, so test the CODE
    kill -0 $pid 2>/dev/null || { echo "__SERVER_DIED__"; return; }
    sleep 2
  done
  [ "$code" = "200" ] || { kill $pid 2>/dev/null; echo "__NOT_READY__"; return; }
  out=$(curl -s "http://127.0.0.1:$PROBE_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"Write a Python function that returns the nth Fibonacci number."}],"temperature":0,"top_p":1,"max_tokens":300}' \
        | "$OMKPY" -c 'import json,sys
d=json.load(sys.stdin); m=d["choices"][0]["message"]
# Reasoning model + --jinja: a 300-token cap never leaves the thinking block, so `content`
# is EMPTY and the text lives in `reasoning_content`. Reading content alone made GATE-M see
# two empty strings and (correctly) refuse. Either field is fine for a DIFFERENCE test.
print(m.get("content") or m.get("reasoning_content") or "")' 2>/dev/null)
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  echo "$out"
}
say "GATE-M: probing override mechanism (greedy, same GGUF, t8 vs t10)"
A=$(probe 8); B=$(probe 10)
la=${#A}; lb=${#B}
say "GATE-M: t8 len=$la  t10 len=$lb"
if [ "$la" -lt 20 ] || [ "$lb" -lt 20 ]; then
  say "REFUSE GATE-M: an arm returned empty/short output (t8=$la t10=$lb) -- INCONCLUSIVE, not a pass"
  exit 3
fi
if [ "$A" = "$B" ]; then
  say "REFUSE GATE-M: t8 and t10 greedy outputs are BYTE-IDENTICAL -- override is inert"
  exit 3
fi
say "GATE-M ok: outputs differ => --override-kv is live on this build"

# ---------------- cells ----------------
for a in "${ARMS[@]}"; do
  IFS="|" read -r B_ CELL TOPK <<<"$a"
  out=$RES/qwen_suite/$B_/$CELL
  [ -f "$out/summary.json" ] && { say "SKIP $B_/$CELL (done)"; continue; }
  [ -d "$out" ] && mv "$out" "${out}_PARTIAL_$(date -u +%Y%m%dT%H%M%SZ)"
  TOK=$TOK_BASE; [ -d "$WORK/armJ" ] && TOK=$WORK/armJ
  EXTRA=""
  if [ "$TOPK" = "10" ]; then
    EXTRA="--metadata backend_args.llama_extra=[\"--override-kv\",\"$OVR\"]"
  fi
  say "===== $B_/$CELL topk=$TOPK"
  "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$B_" --quant q6_k \
      --model "$ARMJ" --tokenizer "$TOK" --served-name "$CELL" --port "$PORT" \
      --results-dir "$RES/qwen_suite" --parallel 2 --gpus 1 \
      --sampler-profile qwen3_6 --sampler recommended \
      --metadata backend_args.llama_ctx=49152 $EXTRA 2>&1 | tail -5
  rc=$?
  L="$out/server.log"
  if [ ! -f "$L" ]; then
    say "FAIL $B_/$CELL: no server.log -- died before serving (rc=$rc); see stdout"
    continue
  fi
  printf '{"intended_topk":%s,"llama_extra":%s,"gate_m":"passed at run start","note":"routing DECLARED + mechanism-gated; server.log carries no hparams (bug-615)"}\n' \
    "$TOPK" "$([ "$TOPK" = 10 ] && echo "[\"--override-kv\",\"$OVR\"]" || echo "[]")" > "$out/routing.json"
  gs=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  gn=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  say "GEOMETRY $B_/$CELL per_slot=${gs:-?} slots=${gn:-?} (want 24576 / 2) rc=$rc"
  fleet=$(ls "$out" 2>/dev/null | grep -c "server.gpu0")
  if [ "${fleet:-0}" -gt 0 ]; then say "ALERT $B_/$CELL: gpu0 server log present -- pin leaked!"; fi
  if [ -f "$out/summary.json" ]; then
    s=$("$OMKPY" -c "import json,sys;d=json.load(open(sys.argv[1]));c=d.get('generation_caps') or {};print(d['score'], c.get('verdict'))" "$out/summary.json" 2>/dev/null)
    say "SCORE $B_/$CELL topk=$TOPK = $s"
  else
    say "FAIL $B_/$CELL: no summary.json"
  fi
done
say "ROUTING_B_DONE"
