#!/usr/bin/env bash
# Re-run HumanEval+ for the two HYBRID arms at the comparators' geometry (24576 / 2 slots).
#
# WHY THIS ONE IS NOT MERELY UNMATCHED, IT IS KNOWN-FATAL.
#   humaneval_full_think asks max_gen_toks=16384 (thinking budget 12288).
#   The hybrid chain served armI/armJ at per-slot n_ctx = 16384.
#   prompt + 16384 > 16384  ->  zero headroom. This is the T172.4 SAT_COLLAPSE condition
#   (per-slot ctx < prompt + max_gen_toks => silent generation collapse), which our own
#   protocol says invalidates the cell. Every other arm ran at 24576 with ~8k spare.
# So armJ's 0.9817 ("best in the program", currently in the HF draft) and armI's 0.9695 are
# both measured under a geometry we have already documented as invalid.
#
# Preserves the original cells; writes <name>_ctx24576. Geometry gate is fatal, not advisory.
set -u
export CUDA_VISIBLE_DEVICES=1
OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
WORK=/mnt/sdc/ream-work; GG=$WORK/gguf; RES=/srv/ml/eval_results/ream_arms
PORT=8099; CTX=24576; PAR=2
say() { echo "[heplus $(date -u +%H:%M:%S)Z] $*"; }

# serialise behind the MPE re-basis -- one job per GPU, always
while pgrep -f "[m]pe_rebasis.sh" >/dev/null 2>&1 || pgrep -f "[o]mk_eval.py" >/dev/null 2>&1; do sleep 60; done
say "GPU1 released, starting"

ran=0; failed=0
for spec in "armI:hybrid_p12_ourssal_reapfloor" "armJ:hybrid_p24_ourssal_reapfloor"; do
  arm=${spec%%:*}; base=${spec#*:}; name="${base}_ctx${CTX}"
  g=$(ls "$GG/${arm}_imat"/*-Q6_K.gguf 2>/dev/null | head -1)
  [ -n "$g" ] && [ -s "$g" ] || { say "SKIP $arm: no Q6_K"; failed=$((failed+1)); continue; }
  [ -f "$RES/humaneval_full_think/$name/summary.json" ] && { say "SKIP $name (exists)"; continue; }
  say "==== $name <- $g (pin llama_ctx=$CTX parallel=$PAR)"
  "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template humaneval_full_think --quant q6_k \
      --model "$g" --tokenizer "$WORK/$arm" --served-name "$name" --port "$PORT" \
      --results-dir "$RES" --parallel "$PAR" --metadata backend_args.llama_ctx=$CTX
  say "<<<< END $name rc=$?"
  L="$RES/humaneval_full_think/$name/server.log"
  got=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  slots=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  say "GEOMETRY $name per_slot=${got:-unknown} slots=${slots:-unknown} (want $CTX/$PAR)"
  if [ "${got:-0}" != "$CTX" ] || [ "${slots:-0}" != "$PAR" ]; then
    say "HEPLUS_ABORT $name: geometry not honoured"; failed=$((failed+1)); continue; fi
  say "GEOMETRY_OK $name"
  s="$RES/humaneval_full_think/$name/summary.json"
  if [ -f "$s" ]; then ran=$((ran+1)); say "SCORE $name = $("$OMKPY" -c "
import json,sys; print(json.load(open(sys.argv[1])).get('score'))" "$s")"
  else say "FAIL $name: no summary"; failed=$((failed+1)); fi
done
say "=== HEPLUS_REBASIS_DONE ran=$ran failed=$failed ==="
