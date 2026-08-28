#!/usr/bin/env bash
# GPQA Diamond (198q) for a3b-gepo1, dropped into the EXISTING qwen_suite cohort
# alongside the banked armJ row.
#
# armJ IS ALREADY DONE. /srv/ml/eval_results/qwen_suite/gpqa_diamond_full/qwenhybridp24_q6k
# = 0.8333 (165/198), sampler=recommended, q6_k, 1.46h, sanity_warnings []. Its GGUF was
# deleted in the 2026-08-24 purge, but the CELL is intact and on exactly the basis this
# script reproduces, so there is nothing to rebuild and nothing to re-run on that arm.
#
# THE COHORT IS `recommended`, NOT GREEDY -- AND THAT IS DELIBERATE.
# eval/models/qwen3_6.yaml marks greedy explicitly NON-VIABLE for this family on open-ended
# thinking benches (256e and 184e both degenerate at temp 0: early-EOS fragments, token
# repetition, ~90k-char runaways). bench_policy.default = recommended (temp 0.6 / top_p 0.95
# / top_k 20 / do_sample true). A greedy GPQA attempt on THIS EXACT MODEL was already made
# and killed on 2026-08-20 -- it survives as qwenhybridp24_q6k_GREEDY_ABORTED_<ts>.
#
# So this row does NOT belong in the same table as today's LCB-48k / MPE-100 rows:
#   ream_arms/*  -> GREEDY  (code benches, reasoning-off, low variance)
#   qwen_suite/* -> recommended temp 0.6  (thinking benches)
# Never pool them. summary.json.sampler.name is the discriminator and is gated below.
#
# NO LLAMA_EXTRA. gate9c_armJ.sh -- the driver that produced the armJ row -- does NOT pass
# --jinja/--chat-template-file; the qwen_suite cohort serves the GGUF's EMBEDDED template.
# The ream_arms LCB/MPE chain DOES pass the fixed jinja. Carrying that flag over from
# eval_gepo1.sh would silently change the prompt basis, so it is explicitly unset here.
#
# GEOMETRY: per_slot 45056 x parallel 2 = 90112 total (llama.cpp divides -c by --parallel,
# bug-597). Copied from the armJ cell's own server.log, and gated on read-back.
#
# SAMPLED RUN => --use_cache IS A NO-OP (do_sample=true). This run is NOT resumable; a death
# restarts from 0. That is inherent to the cohort's sampler, not a missing flag.
#
# GPU1 ONLY.
set -uo pipefail

OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
RES=/srv/ml/eval_results/qwen_suite
GGUF=/mnt/sdc/ml/brevity/gepo/gguf_gepo1/a3b-gepo1-Q6_K.gguf
TOK=/mnt/sdc/ml/brevity/gepo/a3b-gepo1
LOG=/mnt/sdc/ml/brevity/gepo/gpqa_gepo1.log
LCBLOG=/mnt/sdc/ml/brevity/gepo/eval_gepo1.log

NAME=qwena3bgepo1_q6k
REF=qwenhybridp24_q6k          # armJ, banked
B=gpqa_diamond_full
SLOT=45056
PAR=2
TOTAL=$(( SLOT * PAR ))
PROFILE=qwen3_6
SAMPLER=recommended
PORT=8099

export CUDA_VISIBLE_DEVICES=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
export LM_EVAL_BIN=/root/anaconda3/envs/omnimergekit/bin/lm-eval
unset LLAMA_EXTRA            # see header -- the qwen_suite cohort uses the EMBEDDED template
unset LLAMA_ARG_SPEC_TYPE    # no speculative decoding, same as gate9c

exec >>"$LOG" 2>&1
ts(){ date -u '+%F %T UTC'; }
say(){ echo ">>> [$(ts)] $*"; }

echo "=== gpqa_gepo1 start $(ts) ==="

# ---- wait for the LCB/MPE chain to release GPU1 ------------------------------
waited=0
while pgrep -f "eval_gepo1[.]sh" >/dev/null 2>&1; do
  sleep 60; waited=$((waited+60))
  [ $((waited % 900)) -eq 0 ] && say "eval_gepo1 still running (${waited}s)"
  if [ "$waited" -ge 21600 ]; then say "ABORT: eval_gepo1 still running after 6h"; exit 3; fi
done
say "eval_gepo1 finished (waited ${waited}s)"
grep -qa "EVAL_GEPO1_DONE" "$LCBLOG" 2>/dev/null \
  && say "LCB/MPE chain reached its sentinel" \
  || say "WARNING: eval_gepo1 exited WITHOUT EVAL_GEPO1_DONE -- check $LCBLOG"

for _ in $(seq 1 30); do
  free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free" -ge 60000 ] && break
  say "GPU1 only ${free}MiB free, waiting"; sleep 60
done
say "GPU1 free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)MiB"

# ---- preflight ---------------------------------------------------------------
FAIL=0
for p in "$GGUF" "$TOK/tokenizer.json" "$OMK/eval/omk_eval.py" \
         "$OMK/eval/templates/$B.yaml" "$OMK/eval/models/${PROFILE}.yaml"; do
  [ -s "$p" ] || { say "MISSING: $p"; FAIL=1; }
done

# The comparator must exist AND record the sampler we intend. This is the gate that
# would have caught the 2026-08-20 greedy-vs-recommended mismatch before it burned time.
REFS=$RES/$B/$REF/summary.json
if [ ! -f "$REFS" ]; then
  say "REFUSE: comparator $REFS missing -- nothing to compare against"; FAIL=1
else
  got=$("$OMKPY" - "$REFS" <<'PY'
import json,sys
try: print(((json.load(open(sys.argv[1])).get("sampler") or {}).get("name")) or "NONE")
except Exception: print("UNREADABLE")
PY
)
  say "comparator $REF sampler=$got (want $SAMPLER)"
  [ "$got" = "$SAMPLER" ] || { say "REFUSE: comparator not on $SAMPLER"; FAIL=1; }
fi

[ -f "$RES/$B/$NAME/summary.json" ] && { say "SKIP: $NAME already has a summary"; FAIL=2; }
[ "$FAIL" = 0 ] || { say "PREFLIGHT stop (code $FAIL)"; [ "$FAIL" = 2 ] || exit 2; }

# ---- run ---------------------------------------------------------------------
if [ ! -f "$RES/$B/$NAME/summary.json" ]; then
  say "===== $B $NAME  per_slot=$SLOT par=$PAR total=$TOTAL sampler=$SAMPLER (NOT resumable)"
  t0=$SECONDS
  "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$B" --quant q6_k \
      --model "$GGUF" --tokenizer "$TOK" --served-name "$NAME" --port "$PORT" \
      --results-dir "$RES" --parallel "$PAR" \
      --sampler-profile "$PROFILE" --sampler "$SAMPLER" \
      --metadata backend_args.llama_ctx=$TOTAL
  say "<<<< END rc=$? in $(( (SECONDS-t0)/60 ))m"
fi

S=$RES/$B/$NAME/summary.json
L=$RES/$B/$NAME/server.log

# ---- GATE 1: geometry --------------------------------------------------------
got=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
sl=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
say "GEOMETRY per_slot=${got:-unknown} slots=${sl:-unknown} (want $SLOT / $PAR)"
GEO_OK=1
[ "${got:-0}" = "$SLOT" ] && [ "${sl:-0}" = "$PAR" ] || {
  say "!!! GEOMETRY MISMATCH -- this row is NOT comparable to $REF"; GEO_OK=0; }

[ -f "$S" ] || { say "FATAL: no summary.json"; exit 4; }

# ---- GATE 2: sampler provenance ---------------------------------------------
sn=$("$OMKPY" - "$S" <<'PY'
import json,sys
print(((json.load(open(sys.argv[1])).get("sampler") or {}).get("name")) or "NONE")
PY
)
say "SAMPLER recorded=$sn (want $SAMPLER)"
[ "$sn" = "$SAMPLER" ] || say "!!! SAMPLER MISMATCH -- NOT tableable against $REF"

# ---- comparison --------------------------------------------------------------
say "=== GPQA Diamond 198q: a3b-gepo1 vs armJ (cohort: qwen_suite, sampler=recommended) ==="
"$OMKPY" - "$RES/$B/$REF/summary.json" "$S" <<'PY'
import json, sys

def load(p):
    try: return json.load(open(p))
    except Exception: return None

rows = [("armJ", sys.argv[1]), ("gepo1", sys.argv[2])]
print(f"{'arm':<8}{'score':>9}{'n':>6}{'p50 tok':>9}{'p90 tok':>9}{'max':>8}"
      f"{'sum tok':>10}{'trunc':>7}{'empty':>7}{'sampler':>14}{'temp':>6}{'hrs':>7}")
print("-" * 100)
vals = {}
for tag, p in rows:
    d = load(p)
    if d is None:
        print(f"{tag:<8}{'(missing)':>9}"); continue
    sm = d.get("sampler") or {}
    par = sm.get("resolved") if isinstance(sm.get("resolved"), dict) else sm
    t = d.get("token_stats") or {}; c = t.get("completion_tokens") or {}
    fr = t.get("finish_reasons") or {}
    vals[tag] = (d.get("score"), c.get("sum"), c.get("p50"))
    print(f"{tag:<8}{d.get('score'):>9.4f}{t.get('n') or 0:>6}{c.get('p50') or 0:>9}"
          f"{c.get('p90') or 0:>9}{c.get('max') or 0:>8}{c.get('sum') or 0:>10}"
          f"{fr.get('length') or 0:>7}{t.get('empty_completions') or 0:>7}"
          f"{sm.get('name'):>14}{par.get('temperature'):>6}"
          f"{(d.get('duration_s') or 0)/3600:>7.2f}")

if "armJ" in vals and "gepo1" in vals:
    (sa, ta, pa), (sb, tb, pb) = vals["armJ"], vals["gepo1"]
    n = 198
    print()
    print(f"  score   {sa:.4f} -> {sb:.4f}   {(sb-sa)*100:+.2f} pp "
          f"({round(sa*n)} -> {round(sb*n)} of {n} correct, {round(sb*n)-round(sa*n):+d} q)")
    print(f"  tokens  sum {ta} -> {tb}  ({(tb-ta)/ta*100:+.1f}%)   "
          f"p50 {pa} -> {pb}  ({(pb-pa)/max(pa,1)*100:+.1f}%)   <- brevity readout")
    print()
    print("  NOTE: sampled cohort (temp 0.6, do_sample=true). Each cell is ONE draw; a")
    print("        few-question delta is within sampling noise. Only a repeat draw gives")
    print("        the band -- and armJ's GGUF was deleted, so a repeat needs a rebuild")
    print("        from /mnt/sdc/ream-work/armJ + the retained gguf/armJ_imat/imatrix.dat.")
PY

echo "###### GPQA_GEPO1_DONE $(ts) geo_ok=$GEO_OK ######"
