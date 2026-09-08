#!/usr/bin/env bash
# CoderX (armJ) serving-routing decision: top-8 vs top-10 on the axes that can decide it.
#
# WHY NOT MultiPL-E: the GPU1-pinned MPE matrix (2026-08-20) put every config inside
# 0.863-0.900 while armJ's own two top-10 draws spanned 3.7pp. MPE cannot resolve it.
#
# WHY THESE TWO BENCHES: the July finding was that top-10 buys IFEval and costs LCB.
# Banked repeat bands (from what turned out to be a same-config duplicate pair):
#   lcb_v6_77q  0.7013 / 0.6883 -> 1.3pp   <- tight enough to resolve a real effect
#   ifeval_100  0.70   / 0.73   -> 3.0pp
#
# WHY RE-RUN top-8 INSTEAD OF REUSING THE BANKED CELL: banked armJ cells predate the
# GPU1 pin and may have run a 2-GPU fleet. Different serving topology = different
# batching = not a matched basis. Both arms run here, same box, same geometry.
#
# PROVENANCE WARNING THIS SCRIPT EXISTS TO AVOID (bug-610): every GGUF on this box bakes
# qwen35moe.expert_used_count=8. The banked cell named "qwencodermpe_t10_q6k" points at
# the SAME FILE as "qwencodermpe_q6k" with no override -- it was top-8 wearing a t10 name.
# A cell NAME is not a serving config. Hence GATE-W below.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1   # GPU0 is NOT ours. Never rely on --gpus auto.
# PATH: a nohup'd non-interactive shell does NOT get the conda env on PATH. ifeval
# shells out to the `lm-eval` BINARY (unlike multipl_e, which is a native runner), so
# without this every lm-eval-backed cell dies in 13s with FileNotFoundError: 'lm-eval'.
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
WORK=/mnt/sdc/ream-work
RES=/srv/ml/eval_results_routing
PORT=8099
TOK_BASE=/srv/ml/models/Qwen3.6-35B-A3B
ARMJ=$WORK/gguf/armJ_imat/armJ-Q6_K.gguf
LOG=$WORK/run_routing_decision.log
say(){ echo "[route $(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# bench|cell|topk   -- IFEval first (30 min/cell), LCB after (160 min/cell)
ARMS=(
 "ifeval_100|armJ_t8_a|8"
 "ifeval_100|armJ_t10_a|10"
 "lcb_v6_77q|armJ_t8_a|8"
 "lcb_v6_77q|armJ_t10_a|10"
)

[ -s "$ARMJ" ] || { say "REFUSE: missing $ARMJ"; exit 1; }
freeg=$(df -BG --output=avail / | tail -1 | tr -dc "0-9")
[ "${freeg:-0}" -ge 210 ] || { say "REFUSE: root fs ${freeg}G free (floor 200G)"; exit 1; }
g0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
say "preflight ok: ${#ARMS[@]} cells, disk ${freeg}G, GPU0 at ${g0} MiB (must stay 0)"

for a in "${ARMS[@]}"; do
  IFS="|" read -r B CELL TOPK <<<"$a"
  out=$RES/qwen_suite/$B/$CELL
  [ -f "$out/summary.json" ] && { say "SKIP $B/$CELL (done)"; continue; }
  [ -d "$out" ] && mv "$out" "${out}_PARTIAL_$(date -u +%Y%m%dT%H%M%SZ)"
  TOK=$TOK_BASE; [ -d "$WORK/armJ" ] && TOK=$WORK/armJ
  EXTRA=""
  if [ "$TOPK" = "10" ]; then
    EXTRA="--metadata backend_args.llama_extra=[\"--override-kv\",\"qwen35moe.expert_used_count=int:10\"]"
  fi
  say "===== $B/$CELL topk=$TOPK"
  "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$B" --quant q6_k \
      --model "$ARMJ" --tokenizer "$TOK" --served-name "$CELL" --port "$PORT" \
      --results-dir "$RES/qwen_suite" --parallel 2 --gpus 1 \
      --sampler-profile qwen3_6 --sampler recommended \
      --metadata backend_args.llama_ctx=49152 $EXTRA 2>&1 | tail -5
  rc=$?
  L="$out/server.log"
  # A cell that never produced a server.log CRASHED before serving -- report that as the
  # real cause. Running GATE-W first would blame "no --override-kv" for what is actually
  # a dead runner (seen 2026-08-20: lm-eval not on PATH -> 13s death -> bogus GATE-W FAIL).
  if [ ! -f "$L" ]; then
    say "FAIL $B/$CELL: no server.log -- cell died before serving (rc=$rc); see .out"
    continue
  fi
  # GATE-W: the override must be WIRED, not just declared. A t10 cell whose server
  # command line has no --override-kv is a top-8 run wearing a t10 name -> void it.
  if [ "$TOPK" = "10" ]; then
    if grep -aq -- "override-kv" "$L" 2>/dev/null; then
      say "GATE-W ok $B/$CELL: --override-kv present on server cmdline"
    else
      say "GATE-W FAIL $B/$CELL: no --override-kv in server.log -- VOIDING cell"
      mv "$out" "${out}_VOID_NO_OVERRIDE_$(date -u +%Y%m%dT%H%M%SZ)" 2>/dev/null
      continue
    fi
  fi
  gs=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  gn=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  say "GEOMETRY $B/$CELL per_slot=${gs:-?} slots=${gn:-?} (want 24576 / 2) rc=$rc"
  fleet=$(ls "$out" 2>/dev/null | grep -c "server.gpu0")
  if [ "${fleet:-0}" -gt 0 ]; then say "ALERT $B/$CELL: gpu0 server log present -- pin leaked!"; fi
  if [ -f "$out/summary.json" ]; then
    s=$("$OMKPY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['score'], (d.get('sampler') or {}).get('name','?'))" "$out/summary.json" 2>/dev/null)
    say "SCORE $B/$CELL topk=$TOPK = $s"
  else
    say "FAIL $B/$CELL: no summary.json"
  fi
done
say "ROUTING_DECISION_DONE"
