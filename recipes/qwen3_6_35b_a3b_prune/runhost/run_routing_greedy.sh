#!/usr/bin/env bash
# armJ routing decision -- GREEDY REASONING-OFF CODE COHORT (t8 vs t10).
#
# WHY: the sampled cohort (LCB/IFEval @ temp 0.6) cannot resolve this contrast. Measured
# MDE at one draw per arm is +/-9 to +/-14 pp; the observed t8->t10 gap was 5.19 pp, i.e.
# far inside noise (McNemar p=0.52). Buying it down needs k=6-8 draws/arm = 12-16 h.
#
# Greedy is NOT viable for this family on OPEN-ENDED THINKING benches (eval/models/
# qwen3_6.yaml: 256e and the 184e prune both degenerate at temp 0 -- fragments, runaway
# loops). But code benches run reasoning-OFF (omk's llama_bench_defaults injects
# --reasoning off; thinking_budget=0), and greedy completion of short code is stable.
# Measured on this family, this rig, 2026-07-15:
#     temp-0.6 (any parallel) +/-5 pp     | greedy @p2 +/-0.3 pp
#     greedy @p4 +/-0.16 pp  <-- USE THIS | greedy @p8 +/-2.7 pp (batch-numerics drift)
# So a 7-minute greedy MPE cell resolves ~30x tighter than a 2.7-hour sampled LCB cell.
#
# COHORT DISCIPLINE: this is a SEPARATE cohort from draw A/B. summary.json.sampler.name
# will read `template_default` here and `recommended` there. NEVER tabulate them together.
#
# WHAT THIS IS NOT: reasoning-off code completion exercises a different regime than
# thinking-mode LCB. This is an ADDITIONAL point on the routing question, not a
# substitute for the sampled cohort.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1   # GPU0 is NOT ours (96GB and idle is still not ours).
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH   # mbpp shells to lm-eval (bug-612)
export HF_ALLOW_CODE_EVAL=1     # mbpp/HE exec-based scorers refuse without it
OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
WORK=/mnt/sdc/ream-work
RES=/srv/ml/eval_results_routing
PORT=8099
PROBE_PORT=8098
TOK_BASE=/srv/ml/models/Qwen3.6-35B-A3B
ARMJ=$WORK/gguf/armJ_imat/armJ-Q6_K.gguf
LSRV=/opt/llama.cpp/build/bin/llama-server
LOG=$WORK/run_routing_greedy.log
OVR="qwen35moe.expert_used_count=int:10"
BLOG=$WORK/run_routing_b.log
# 98304 total / 4 slots = 24576 per slot. mbpp asks max_gen_toks=16384; a per-slot ctx
# BELOW that is the T172.4 SAT_COLLAPSE trap. llama_ctx is TOTAL, not per-slot -- pin it
# and read the per-slot value back from server.log (never assume).
CTX=98304
PAR=4                            # the measured-band parallel. Same for EVERY cell so the
                                 # uniform batch-numerics offset cancels in the contrast.
say(){ echo "[greedy $(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# bench | cell | topk
ARMS=(
 "multipl_e_100|armJ_g_t8|8"
 "multipl_e_100|armJ_g_t8r|8"     # REPEAT of the SAME config -> the band on THIS artifact
 "multipl_e_100|armJ_g_t10|10"
 "mbpp_full|armJ_g_t8|8"
 "mbpp_full|armJ_g_t10|10"
)

# ---------------- wait for draw B to release GPU1 ----------------
say "waiting for ROUTING_B_DONE before touching GPU1"
for i in $(seq 1 720); do            # 720 * 60s = 12h ceiling
  grep -q "ROUTING_B_DONE" "$BLOG" 2>/dev/null && { say "draw B finished"; break; }
  sleep 60
done
if ! grep -q "ROUTING_B_DONE" "$BLOG" 2>/dev/null; then
  say "REFUSE: draw B never signalled ROUTING_B_DONE within 12h -- not racing it for the GPU"
  exit 1
fi
# belt: no llama-server may still hold the card. `llama[-]server` splits the pattern so
# pgrep cannot self-match its own argv (bug-614 -- same property that makes pkill -f fatal).
for i in $(seq 1 30); do
  pgrep -f "llama[-]server" >/dev/null 2>&1 || break
  say "  a llama-server is still alive; waiting"
  sleep 20
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
say "GPU1 at ${used} MiB, GPU0 at $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) MiB"

# ---------------- preflight ----------------
[ -s "$ARMJ" ] || { say "REFUSE: missing $ARMJ"; exit 1; }
[ -x "$LSRV" ] || { say "REFUSE: missing $LSRV"; exit 1; }
freeg=$(df -BG --output=avail / | tail -1 | tr -dc "0-9")
[ "${freeg:-0}" -ge 210 ] || { say "REFUSE: root fs ${freeg}G free (floor 200G)"; exit 1; }
# Doctrine: confirm the frozen generation block is STILL greedy before launching. A sampled
# run never edits a template -- it passes --sampler. Any drift here = STOP.
for t in multipl_e_100 mbpp_full; do
  g=$("$OMKPY" - "$OMK/eval/templates/$t.yaml" <<'PY'
import sys, yaml
gen = (yaml.safe_load(open(sys.argv[1])) or {}).get("generation") or {}
ok = (gen.get("temperature") == 0.0 and gen.get("do_sample") is False
      and gen.get("top_p") == 1.0 and gen.get("top_k") == 0)
print("GREEDY" if ok else "DRIFT temp=%s do_sample=%s top_p=%s top_k=%s" % (
    gen.get("temperature"), gen.get("do_sample"), gen.get("top_p"), gen.get("top_k")))
PY
)
  say "template $t: $g"
  [ "$g" = "GREEDY" ] || { say "REFUSE: $t is not frozen greedy -- fix the template, then relaunch"; exit 2; }
done
say "preflight ok: ${#ARMS[@]} cells, disk ${freeg}G, parallel=$PAR ctx=$CTX"

# ---------------- GATE-M: is the override mechanism live on this build? ----------------
probe(){   # $1 = topk ; echoes the completion text
  local topk="$1" extra=() pid out code i
  [ "$topk" = "10" ] && extra=(--override-kv "$OVR")
  "$LSRV" -m "$ARMJ" --port "$PROBE_PORT" -c 4096 -ngl 99 --jinja --no-warmup \
      "${extra[@]}" > "$WORK/gprobe_t${topk}.log" 2>&1 &
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
# is EMPTY and the text lives in `reasoning_content`. Either field works for a DIFF test.
print(m.get("content") or m.get("reasoning_content") or "")' 2>/dev/null)
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  echo "$out"
}
say "GATE-M: probing override mechanism (t8 vs t10 on the same GGUF)"
A=$(probe 8); B=$(probe 10); la=${#A}; lb=${#B}
say "GATE-M: t8 len=$la  t10 len=$lb"
if [ "$la" -lt 20 ] || [ "$lb" -lt 20 ]; then
  say "REFUSE GATE-M: an arm returned empty/short output (t8=$la t10=$lb) -- INCONCLUSIVE, not a pass"
  exit 3
fi
[ "$A" = "$B" ] && { say "REFUSE GATE-M: t8/t10 outputs BYTE-IDENTICAL -- override inert"; exit 3; }
say "GATE-M ok: outputs differ => --override-kv is live"

# ---------------- cells ----------------
first_score=""
for a in "${ARMS[@]}"; do
  IFS="|" read -r B_ CELL TOPK <<<"$a"
  out=$RES/qwen_suite/$B_/$CELL
  [ -f "$out/summary.json" ] && { say "SKIP $B_/$CELL (done)"; continue; }
  [ -d "$out" ] && mv "$out" "${out}_PARTIAL_$(date -u +%Y%m%dT%H%M%SZ)"
  TOK=$TOK_BASE; [ -d "$WORK/armJ" ] && TOK=$WORK/armJ
  EXTRA=""
  [ "$TOPK" = "10" ] && EXTRA="--metadata backend_args.llama_extra=[\"--override-kv\",\"$OVR\"]"
  say "===== $B_/$CELL topk=$TOPK (greedy: NO --sampler flag, template default)"
  # NOTE: no --sampler / --sampler-profile. The frozen template block IS greedy; passing a
  # sampler flag here is what would corrupt the cohort. --served-name must be $CELL or omk
  # names the results dir after the model and the gates read the wrong place (bug-613).
  "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$B_" --quant q6_k \
      --model "$ARMJ" --tokenizer "$TOK" --served-name "$CELL" --port "$PORT" \
      --results-dir "$RES/qwen_suite" --parallel "$PAR" --gpus 1 \
      --metadata backend_args.llama_ctx=$CTX $EXTRA 2>&1 | tail -5
  rc=$?
  L="$out/server.log"
  if [ ! -f "$L" ]; then
    say "FAIL $B_/$CELL: no server.log -- died before serving (rc=$rc)"
    continue
  fi
  printf '{"intended_topk":%s,"llama_extra":%s,"sampler":"template_default (greedy)","parallel":%s,"llama_ctx":%s,"gate_m":"passed at run start","note":"greedy reasoning-off code cohort; NOT comparable to the temp-0.6 draw A/B cells"}\n' \
    "$TOPK" "$([ "$TOPK" = 10 ] && echo "[\"--override-kv\",\"$OVR\"]" || echo "[]")" \
    "$PAR" "$CTX" > "$out/routing.json"
  gs=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  gn=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  say "GEOMETRY $B_/$CELL per_slot=${gs:-?} slots=${gn:-?} (want 24576 / $PAR) rc=$rc"
  ls "$out" 2>/dev/null | grep -q "server.gpu0" && say "ALERT $B_/$CELL: gpu0 server log present -- pin leaked!"
  if [ -f "$out/summary.json" ]; then
    s=$("$OMKPY" -c "
import json,sys
d=json.load(open(sys.argv[1])); sm=d.get('sampler') or {}; c=d.get('generation_caps') or {}
print('%s sampler=%s verdict=%s'%(d['score'], sm.get('name'), c.get('verdict')))" "$out/summary.json")
    say "SCORE $B_/$CELL topk=$TOPK = $s"
    # Cheap early gate: the banked temp-0.6 MPE cell read 0.80. Greedy should land at or
    # above that band. A collapse means chat-template / code-extraction breakage, not a
    # routing effect -- stop before spending the remaining ~90 min.
    if [ -z "$first_score" ] && [ "$B_" = "multipl_e_100" ]; then
      first_score=$(echo "$s" | awk '{print $1}')
      bad=$("$OMKPY" -c "import sys;print('1' if float(sys.argv[1])<0.60 else '0')" "$first_score")
      [ "$bad" = "1" ] && { say "REFUSE: first greedy MPE cell = $first_score, far below the 0.80 banked band -- extraction/template problem, NOT routing. Stopping."; exit 4; }
    fi
  else
    say "FAIL $B_/$CELL: no summary.json"
  fi
done
say "ROUTING_GREEDY_DONE"
