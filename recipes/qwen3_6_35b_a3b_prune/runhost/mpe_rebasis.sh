#!/usr/bin/env bash
# Re-run MultiPL-E for the two HYBRID arms on the SAME server geometry as the other eleven.
#
# WHY. The 2026-08-19 hybrid chain served armI/armJ at n_ctx=4096 while every arm they are
# compared against ran at 12288 (4 slots in both cases). The ctx was auto-planned from free
# VRAM, not pinned, so it silently drifted between chains. Prompt truncation is ruled out
# (max MPE prompt = 778 tok, "truncated = 0"), so this is not a known-fatal geometry -- but
# it IS an unmatched axis under a cross-arm comparison, and the MPE deficit that difference
# sits on is the number about to be published. So measure it instead of arguing about it.
#
# PRESERVATION. Writes to NEW cell names (<name>_ctx12288). The original cells are evidence
# and are never overwritten or deleted.
#
# GATE. After each launch the actual per-slot n_ctx is read back out of the server log and
# compared to the pin. A run that did not get the geometry it asked for is aborted, not
# silently banked -- that is the whole point of the exercise.
set -u
export CUDA_VISIBLE_DEVICES=1          # bs2 GPU1 is ours; GPU0 is NOT. Export, never poll.

OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
WORK=/mnt/sdc/ream-work
GG=$WORK/gguf
RES=/srv/ml/eval_results/ream_arms
PORT=8099
CTX=12288
PAR=4

say() { echo "[rebasis $(date -u +%H:%M:%S)Z] $*"; }

ran=0; failed=0
for spec in "armI:hybrid_p12_ourssal_reapfloor" "armJ:hybrid_p24_ourssal_reapfloor"; do
  arm=${spec%%:*}; base=${spec#*:}
  name="${base}_ctx${CTX}"
  g=$(ls "$GG/${arm}_imat"/*-Q6_K.gguf 2>/dev/null | head -1)
  if [ -z "$g" ] || [ ! -s "$g" ]; then say "SKIP $arm: no Q6_K"; failed=$((failed+1)); continue; fi
  if [ -f "$RES/multipl_e_100/$name/summary.json" ]; then say "SKIP $name (exists)"; continue; fi

  say "==== $name  <- $g  (pinning llama_ctx=$CTX, parallel=$PAR)"
  "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template multipl_e_100 --quant q6_k \
      --model "$g" --tokenizer "$WORK/$arm" --served-name "$name" --port "$PORT" \
      --results-dir "$RES" --parallel "$PAR" \
      --metadata backend_args.llama_ctx=$CTX
  rc=$?
  say "<<<< END $name rc=$rc"

  # ---- GEOMETRY GATE: did we actually get what we pinned? ----
  L="$RES/multipl_e_100/$name/server.log"
  got=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  slots=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  say "GEOMETRY $name per_slot_n_ctx=${got:-unknown} n_slots=${slots:-unknown} (want $CTX / $PAR)"
  if [ "${got:-0}" != "$CTX" ] || [ "${slots:-0}" != "$PAR" ]; then
    say "REBASIS_ABORT $name: geometry not honoured — this cell does NOT restore the basis."
    failed=$((failed+1)); continue
  fi
  say "GEOMETRY_OK $name"

  s="$RES/multipl_e_100/$name/summary.json"
  if [ -f "$s" ]; then
    ran=$((ran+1))
    say "SCORE $name = $("$OMKPY" -c "
import json,sys
d=json.load(open(sys.argv[1])); print(d.get('score'))" "$s")"
    say "EMPTIES $name = $("$OMKPY" -c "
import json,sys
n=0
for line in open(sys.argv[1]):
    line=line.strip()
    if line:
        d=json.loads(line)
        if not (d.get('completion') or '').strip(): n+=1
print(n)" "$RES/multipl_e_100/$name/mpe_result.samples.jsonl")/300"
  else
    say "FAIL $name: no summary.json"; failed=$((failed+1))
  fi
done
say "=== MPE_REBASIS_DONE ran=$ran failed=$failed ==="
