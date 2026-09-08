#!/usr/bin/env bash
# GAP-FILL for run_routing_greedy.sh -- reruns any cell that has no summary.json.
#
# WHY THIS EXISTS (bug-617): run_routing_greedy.sh tore down its GATE-M probe server and
# handed off to omk_eval in the SAME SECOND. GPU1 read 0 MiB but utilisation was still 42%
# from the teardown, and omk's gpu_planner refuses at util >= 15%:
#     "REFUSING to launch -- no GPU meets THRESHOLD (need~21267MiB free + util<0.15)"
#     FATAL exit=8
# multipl_e_100/armJ_g_t8 died before serving. The next cell launched ~0s later and
# SUCCEEDED, which is the tell that this is a race, not a config fault -- and exactly why
# it must be fixed with a PREDICATE, not a sleep: the settle time is not a known constant.
#
# The original script is NOT edited (it was running). This one adds wait_gpu_settle() and
# calls it after GATE-M and before EVERY cell, then relies on the existing
# skip-if-summary.json logic to fill only the holes.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
export HF_ALLOW_CODE_EVAL=1
OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
WORK=/mnt/sdc/ream-work
RES=/srv/ml/eval_results_routing
PORT=8099
TOK_BASE=/srv/ml/models/Qwen3.6-35B-A3B
ARMJ=$WORK/gguf/armJ_imat/armJ-Q6_K.gguf
LOG=$WORK/run_routing_greedy_fix.log
GLOG=$WORK/run_routing_greedy.log
OVR="qwen35moe.expert_used_count=int:10"
CTX=98304
PAR=4
say(){ echo "[gfix $(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

ARMS=(
 "multipl_e_100|armJ_g_t8|8"
 "multipl_e_100|armJ_g_t8r|8"
 "multipl_e_100|armJ_g_t10|10"
 "mbpp_full|armJ_g_t8|8"
 "mbpp_full|armJ_g_t10|10"
)

# The predicate the original script was missing. Poll until the card is ACTUALLY idle --
# both free memory AND util below omk's own threshold -- rather than assuming teardown is
# instantaneous. Returns non-zero on timeout so the caller can refuse instead of racing.
wait_gpu_settle(){
  local i used util
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1)
    if [ "${used:-99999}" -lt 2000 ] && [ "${util:-100}" -lt 10 ]; then
      say "  GPU1 settled (mem=${used}MiB util=${util}%) after $((i*5))s"
      return 0
    fi
    sleep 5
  done
  say "  GPU1 did NOT settle in 300s (mem=${used}MiB util=${util}%)"
  return 1
}

say "waiting for ROUTING_GREEDY_DONE before touching GPU1"
for i in $(seq 1 720); do
  grep -q "ROUTING_GREEDY_DONE" "$GLOG" 2>/dev/null && { say "main greedy run finished"; break; }
  sleep 60
done
grep -q "ROUTING_GREEDY_DONE" "$GLOG" 2>/dev/null || { say "REFUSE: main run never signalled -- not racing it"; exit 1; }
for i in $(seq 1 30); do
  pgrep -f "llama[-]server" >/dev/null 2>&1 || break   # bug-614: split pattern, no self-match
  say "  a llama-server is still alive; waiting"
  sleep 20
done

missing=0
for a in "${ARMS[@]}"; do
  IFS="|" read -r B_ CELL TOPK <<<"$a"
  [ -f "$RES/qwen_suite/$B_/$CELL/summary.json" ] || { missing=$((missing+1)); say "HOLE: $B_/$CELL"; }
done
[ "$missing" -eq 0 ] && { say "no holes -- nothing to do"; say "ROUTING_GFIX_DONE"; exit 0; }
say "$missing hole(s) to fill"

freeg=$(df -BG --output=avail / | tail -1 | tr -dc "0-9")
[ "${freeg:-0}" -ge 210 ] || { say "REFUSE: root fs ${freeg}G free (floor 200G)"; exit 1; }
# GATE-M is NOT repeated here: the main run already proved --override-kv live on this build
# at 06:13:43Z (t8 len=1091 vs t10 len=1043, differ), same binary, same GGUF, same session.
# Re-probing would only re-introduce the teardown race this script exists to fix.
say "GATE-M inherited from the main run (t8 1091 / t10 1043, differ) -- not re-probed"

for a in "${ARMS[@]}"; do
  IFS="|" read -r B_ CELL TOPK <<<"$a"
  out=$RES/qwen_suite/$B_/$CELL
  [ -f "$out/summary.json" ] && { say "SKIP $B_/$CELL (done)"; continue; }
  [ -d "$out" ] && mv "$out" "${out}_PARTIAL_$(date -u +%Y%m%dT%H%M%SZ)"
  wait_gpu_settle || { say "REFUSE $B_/$CELL: GPU never settled"; continue; }
  TOK=$TOK_BASE; [ -d "$WORK/armJ" ] && TOK=$WORK/armJ
  EXTRA=""
  [ "$TOPK" = "10" ] && EXTRA="--metadata backend_args.llama_extra=[\"--override-kv\",\"$OVR\"]"
  say "===== $B_/$CELL topk=$TOPK (greedy, no --sampler)"
  "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$B_" --quant q6_k \
      --model "$ARMJ" --tokenizer "$TOK" --served-name "$CELL" --port "$PORT" \
      --results-dir "$RES/qwen_suite" --parallel "$PAR" --gpus 1 \
      --metadata backend_args.llama_ctx=$CTX $EXTRA 2>&1 | tail -5
  rc=$?
  L="$out/server.log"
  [ -f "$L" ] || { say "FAIL $B_/$CELL: no server.log -- died before serving (rc=$rc)"; continue; }
  printf '{"intended_topk":%s,"llama_extra":%s,"sampler":"template_default (greedy)","parallel":%s,"llama_ctx":%s,"gate_m":"inherited from main run 06:13:43Z","note":"greedy reasoning-off code cohort; NOT comparable to the temp-0.6 draw A/B cells"}\n' \
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
  else
    say "FAIL $B_/$CELL: no summary.json"
  fi
done
say "ROUTING_GFIX_DONE"
