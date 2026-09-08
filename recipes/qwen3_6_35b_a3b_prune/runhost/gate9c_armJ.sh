#!/usr/bin/env bash
# CoderX gate 1/2 (v3): armJ ONLY, dropped into the EXISTING qwen_suite cohort.
#
# WHY v3. v2 (gate9b) matched the cohort's SERVER GEOMETRY but not its SAMPLER. Every
# qwencodermpe_q6k cell records sampler=recommended (qwen3_6 profile: temp 0.6 / top_p 0.95
# / top_k 20 / do_sample=true); v2 ran template_default = GREEDY. A greedy row cannot be
# tabled against a `recommended` cohort, and for this family the profile marks greedy
# explicitly non-viable (256e and 184e both degenerate at temp 0). v2's partial gpqa cell
# was preserved as *_GREEDY_ABORTED_<ts>, not deleted.
#
# NOTE the two cohorts are deliberately different and must never be merged:
#   ream_arms/*        -> GREEDY, 13 arms, the internal A/B that produced the hybrid result.
#   qwen_suite/*       -> recommended (temp 0.6), the PUBLISHED comparison table.
# armJ needs a row in the second one. That is what this script makes.
#
# TWO GATES, both fatal per cell:
#   1. GEOMETRY  -- per-slot n_ctx + n_slots read back from server.log must equal the value
#                   copied from the comparator's own server.log. total = per_slot * parallel
#                   (llama.cpp divides -c by --parallel; bug-597).
#   2. SAMPLER   -- summary.json.sampler.name must equal the comparator cell's recorded
#                   sampler.name. Checked BEFORE the run (comparator readable + is what we
#                   intend) and AFTER (we actually got it). This is the v2 failure, gated.
#
# ============================ STOP-WORD COLLISION (v4, 2026-08-20) ============================
# math500_100 and aime_30 are REMOVED from this chain. Both resolve an `until` containing a
# plain-text delimiter -- minerva_math500 => ["Problem:"], aime24_chat => ["Question:", ...] --
# and llama.cpp matches stop words on the RAW token stream, BEFORE --reasoning-format deepseek
# splits <think> into reasoning_content. Qwen3.x opens nearly every answer with an enumerated
# plan whose first item is "**Understand the Problem:**" / "**Understand the Question:**", so
# the stop fires INSIDE the thinking block and kills the whole generation ~65 chars in.
#
# MEASURED on the math500_100 cell this chain already produced:
#   armJ  reasoning p50=65 chars / content p50=0 / 80 of 100 rows under 100 chars -> 0.2200
#   pub   reasoning p50=0 (skips thinking on 63/100)                              -> 0.7000
#   0 of 200 completions across BOTH arms contain the delimiter anywhere; the 80 short armJ
#   rows collapse to 5 distinct prefixes, each breaking exactly where the delimiter would go.
#   Nothing capped (OMK_CAP_CHECK CLEAN, 0/100), nothing empty. It is the stop, not the model.
# A matched-but-defective basis is still invalid: it differentially penalises whichever arm
# thinks more, which is precisely the axis under test. BOTH cells are void, not just armJ's.
#
# Fix = task variants that drop the delimiter (T177 precedent: minerva_math500_qwen), run for
# BOTH arms by gate9d. This is a NEW basis; never pool with the old cells.
#
# aime_30 was ALSO 45056/slot against max_gen_toks=65536 -- SAT_COLLAPSE-shaped for the whole
# cohort. aime_30_qwen pins 69632/slot so the slot finally exceeds the generation budget.
set -u
export CUDA_VISIBLE_DEVICES=1          # bs2 GPU1 is ours; GPU0 is NOT. Export, never poll.
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
command -v lm-eval >/dev/null || { echo "REFUSING: lm-eval not on PATH"; exit 1; }

OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
WORK=/mnt/sdc/ream-work
RES=/srv/ml/eval_results/qwen_suite
PORT=8099
G=$WORK/gguf/armJ_imat/armJ-Q6_K.gguf
TOK=$WORK/armJ
NAME=qwenhybridp24_q6k
REF=qwencodermpe_q6k                    # the published cut, already evaluated
PROFILE=qwen3_6
SAMPLER=recommended

say(){ echo "[gate9c $(date -u +%H:%M:%S)Z] $*"; }
[ -s "$G" ] || { echo "REFUSING: missing $G"; exit 1; }

# bench | per_slot | parallel   (copied from the REF cell's own server.log)
JOBS=(
  "gpqa_diamond_full|45056|2"
  "gsm8k_100_boxed|45056|2"
  "ifeval_100|24576|2"
  "humaneval_full_think|24576|2"
  "multipl_e_100|24576|2"
  "lcb_v6_77q|45056|2"
)
# v4 2026-08-20: math500_100 and aime_30 REMOVED from this chain -- both are
# STOP-WORD-POISONED for Qwen and are re-run, BOTH ARMS, by gate9d on
# math500_100_qwen / aime_30_qwen. See the "STOP-WORD COLLISION" block above.

# ---- PRE-FLIGHT: every comparator cell must exist and record the sampler we intend ----
say "preflight: verifying all ${#JOBS[@]} comparator cells record sampler=$SAMPLER"
bad=0
for j in "${JOBS[@]}"; do
  B=${j%%|*}
  got=$("$OMKPY" - "$RES/$B/$REF/summary.json" <<'PY' 2>/dev/null
import json,sys
try:
    print(((json.load(open(sys.argv[1])).get("sampler") or {}).get("name")) or "NONE")
except Exception:
    print("UNREADABLE")
PY
)
  if [ "$got" != "$SAMPLER" ]; then say "PREFLIGHT_BAD $B: comparator sampler=$got want=$SAMPLER"; bad=$((bad+1));
  else say "  ok $B comparator sampler=$got"; fi
done
[ "$bad" -eq 0 ] || { say "GATE9C_REFUSE: $bad comparator cell(s) not on $SAMPLER -- no run is comparable"; exit 2; }

ran=0; failed=0; aborted=0
for j in "${JOBS[@]}"; do
  B=${j%%|*}; r=${j#*|}; SLOT=${r%%|*}; PAR=${r##*|}; TOTAL=$(( SLOT * PAR ))
  [ -f "$RES/$B/$NAME/summary.json" ] && { say "SKIP $B (exists)"; continue; }
  say "===== $B  per_slot=$SLOT par=$PAR total=$TOTAL sampler=$SAMPLER"
  "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$B" --quant q6_k \
      --model "$G" --tokenizer "$TOK" --served-name "$NAME" --port "$PORT" \
      --results-dir "$RES" --parallel "$PAR" \
      --sampler-profile "$PROFILE" --sampler "$SAMPLER" \
      --metadata backend_args.llama_ctx=$TOTAL
  say "<<<< END $B rc=$?"

  # ---- GATE 1: geometry ----
  L="$RES/$B/$NAME/server.log"
  got=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  sl=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  say "GEOMETRY $B per_slot=${got:-unknown} slots=${sl:-unknown} (want $SLOT / $PAR)"
  if [ "${got:-0}" != "$SLOT" ] || [ "${sl:-0}" != "$PAR" ]; then
    say "GATE9C_ABORT $B: geometry not honoured -- NOT comparable to the cohort"
    aborted=$((aborted+1)); continue
  fi

  s="$RES/$B/$NAME/summary.json"
  [ -f "$s" ] || { say "FAIL $B: no summary.json"; failed=$((failed+1)); continue; }

  # ---- GATE 2: sampler provenance (the v2 failure) ----
  sn=$("$OMKPY" - "$s" <<'PY'
import json,sys
print(((json.load(open(sys.argv[1])).get("sampler") or {}).get("name")) or "NONE")
PY
)
  say "SAMPLER $B recorded=$sn (want $SAMPLER)"
  if [ "$sn" != "$SAMPLER" ]; then
    say "GATE9C_ABORT $B: sampler mismatch -- this row is NOT tableable against $REF"
    aborted=$((aborted+1)); continue
  fi

  ran=$((ran+1))
  say "SCORE $B armJ = $("$OMKPY" - "$s" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(d.get("score"), d.get("metric"), d.get("filter"))
PY
)"
done
say "=== GATE9C_DONE ran=$ran failed=$failed aborted=$aborted ==="
