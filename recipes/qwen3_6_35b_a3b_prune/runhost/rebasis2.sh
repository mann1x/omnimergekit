#!/usr/bin/env bash
# Re-basis v2 -- MPE + HumanEval+ for the two hybrid arms at the COMPARATORS' per-slot geometry.
#
# WHAT v1 GOT WRONG. `llama_ctx` is the TOTAL KV budget; llama.cpp divides it by --parallel.
# v1 pinned llama_ctx=12288 with 4 slots and got 3072/slot -- worse than the 4096 the hybrids
# already had. The comparators' 12288/slot came from a total of 49152. So pin the TOTAL and
# gate on the PER-SLOT readback.
#
#   MPE : total 49152 / 4 slots = 12288 per slot   (comparators: 12288; hybrids ran 4096)
#   HE+ : total 49152 / 2 slots = 24576 per slot   (comparators: 24576; hybrids ran 16384)
#
# HE+ is the one that is not merely unmatched but KNOWN-FATAL: humaneval_full_think asks
# max_gen_toks=16384 and the hybrids were served 16384 per slot, so prompt+generation cannot
# fit -- the T172.4 SAT_COLLAPSE condition. armJ's 0.9817 sits on that geometry.
#
# v1 also died on HE+ with FileNotFoundError: 'lm-eval' -- setsid/nohup bash carried no env
# PATH. Export it explicitly.
#
# Cells are named by PER-SLOT ctx (unambiguous). Originals and the aborted v1 cells are
# evidence and are never overwritten or deleted.
set -u
export CUDA_VISIBLE_DEVICES=1          # bs2 GPU1 is ours; GPU0 is NOT.
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
command -v lm-eval >/dev/null || { echo "REFUSING: lm-eval not on PATH"; exit 1; }

OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
WORK=/mnt/sdc/ream-work; GG=$WORK/gguf; RES=/srv/ml/eval_results/ream_arms
PORT=8099; TOTAL=49152
say() { echo "[rebasis2 $(date -u +%H:%M:%S)Z] $*"; }
say "lm-eval = $(command -v lm-eval)"

ran=0; failed=0
# bench template par want_per_slot
for job in "multipl_e_100:4:12288" "humaneval_full_think:2:24576"; do
  T=${job%%:*}; rest=${job#*:}; PAR=${rest%%:*}; WANT=${rest#*:}
  for spec in "armI:hybrid_p12_ourssal_reapfloor" "armJ:hybrid_p24_ourssal_reapfloor"; do
    arm=${spec%%:*}; base=${spec#*:}; name="${base}_slot${WANT}"
    g=$(ls "$GG/${arm}_imat"/*-Q6_K.gguf 2>/dev/null | head -1)
    [ -n "$g" ] && [ -s "$g" ] || { say "SKIP $arm/$T: no Q6_K"; failed=$((failed+1)); continue; }
    [ -f "$RES/$T/$name/summary.json" ] && { say "SKIP $T/$name (exists)"; continue; }
    say "==== $T / $name  (total=$TOTAL par=$PAR -> want ${WANT}/slot)"
    "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$T" --quant q6_k \
        --model "$g" --tokenizer "$WORK/$arm" --served-name "$name" --port "$PORT" \
        --results-dir "$RES" --parallel "$PAR" --metadata backend_args.llama_ctx=$TOTAL
    say "<<<< END $T/$name rc=$?"
    L="$RES/$T/$name/server.log"
    got=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
    slots=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
    say "GEOMETRY $T/$name per_slot=${got:-unknown} slots=${slots:-unknown} (want $WANT / $PAR)"
    if [ "${got:-0}" != "$WANT" ] || [ "${slots:-0}" != "$PAR" ]; then
      say "REBASIS_ABORT $T/$name: geometry not honoured -- cell does NOT restore the basis"
      failed=$((failed+1)); continue; fi
    say "GEOMETRY_OK $T/$name"
    s="$RES/$T/$name/summary.json"
    if [ -f "$s" ]; then ran=$((ran+1))
      say "SCORE $T/$name = $("$OMKPY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('score'))" "$s")"
      sj="$RES/$T/$name/mpe_result.samples.jsonl"
      [ -f "$sj" ] && say "EMPTIES $name = $("$OMKPY" -c "
import json,sys
n=0
for l in open(sys.argv[1]):
    l=l.strip()
    if l and not (json.loads(l).get('completion') or '').strip(): n+=1
print(n)" "$sj")/300"
    else say "FAIL $T/$name: no summary"; failed=$((failed+1)); fi
  done
done
say "=== REBASIS2_DONE ran=$ran failed=$failed ==="
