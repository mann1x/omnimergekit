#!/usr/bin/env bash
# CoderX gate 1/2 (v2): armJ ONLY, run into the EXISTING qwen_suite cohort.
#
# WHY NO COMPARATOR RUN. qwencodermpe_q6k already holds the published cut on all these
# benches (same gguf as ream_arms/pub184e_imat). Re-running it would cost 2x and, at my
# earlier invented geometry, would not even have matched. Instead armJ is pinned to each
# bench's EXISTING per-slot value so it drops straight into the published table.
#
# GEOMETRY IS COPIED FROM THE COMPARATOR CELL, not derived. total = per_slot * parallel.
# Readback gate FATAL. (aime_30 runs at 45056/slot against max_gen_toks=65536 -- that is a
# SAT_COLLAPSE-shaped cell for the WHOLE cohort, uniform so comparable, but the absolute
# aime number is suspect for everyone. Flagged, not silently used.)
set -u
export CUDA_VISIBLE_DEVICES=1
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
command -v lm-eval >/dev/null || { echo "REFUSING: lm-eval not on PATH"; exit 1; }
OMK=/srv/ml/repos/omnimergekit; OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
WORK=/mnt/sdc/ream-work; RES=/srv/ml/eval_results/qwen_suite; PORT=8099
G=$WORK/gguf/armJ_imat/armJ-Q6_K.gguf; TOK=$WORK/armJ; NAME=qwenhybridp24_q6k
say(){ echo "[gate9b $(date -u +%H:%M:%S)Z] $*"; }
[ -s "$G" ] || { echo "REFUSING: missing $G"; exit 1; }

# bench | per_slot | parallel   (copied from qwencodermpe_q6k server.logs)
JOBS=(
  "gpqa_diamond_full|45056|2"
  "math500_100|45056|2"
  "gsm8k_100_boxed|45056|2"
  "ifeval_100|24576|2"
  "aime_30|45056|2"
  "humaneval_full_think|24576|2"
  "multipl_e_100|24576|2"
  "lcb_v6_77q|45056|2"
)
ran=0; failed=0; aborted=0
for j in "${JOBS[@]}"; do
  B=${j%%|*}; r=${j#*|}; SLOT=${r%%|*}; PAR=${r##*|}; TOTAL=$(( SLOT * PAR ))
  [ -f "$RES/$B/$NAME/summary.json" ] && { say "SKIP $B (exists)"; continue; }
  say "===== $B  per_slot=$SLOT par=$PAR total=$TOTAL"
  "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$B" --quant q6_k \
      --model "$G" --tokenizer "$TOK" --served-name "$NAME" --port "$PORT" \
      --results-dir "$RES" --parallel "$PAR" --metadata backend_args.llama_ctx=$TOTAL
  say "<<<< END $B rc=$?"
  L="$RES/$B/$NAME/server.log"
  got=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  sl=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  say "GEOMETRY $B per_slot=${got:-unknown} slots=${sl:-unknown} (want $SLOT / $PAR)"
  if [ "${got:-0}" != "$SLOT" ] || [ "${sl:-0}" != "$PAR" ]; then
    say "GATE9B_ABORT $B: geometry not honoured -- NOT comparable to the cohort"
    aborted=$((aborted+1)); continue; fi
  s="$RES/$B/$NAME/summary.json"
  if [ -f "$s" ]; then ran=$((ran+1))
    say "SCORE $B armJ = $("$OMKPY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('score'),d.get('metric'),d.get('filter'))" "$s")"
  else say "FAIL $B: no summary.json"; failed=$((failed+1)); fi
done
say "=== GATE9B_DONE ran=$ran failed=$failed aborted=$aborted ==="
