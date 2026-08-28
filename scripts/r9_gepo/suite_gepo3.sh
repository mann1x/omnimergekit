#!/usr/bin/env bash
# Full omk canonical suite for a3b-gepo3 (R9 run2), dropped into the EXISTING qwen_suite
# cohort alongside the banked armJ (qwenhybridp24_q6k) and gepo1 (qwena3bgepo1_q6k) rows.
#
# CLONE OF suite_gepo1.sh. Three deliberate differences, everything else byte-identical:
#   1. GGUF/TOK/NAME point at gepo3.
#   2. gpqa_diamond_full IS in JOBS. suite_gepo1.sh omitted it because gepo1's GPQA cell
#      had already been produced by gpqa_gepo1.sh that morning; gepo3 has no such cell, so
#      it must run here. Geometry 45056/2 read back from BOTH the gepo1 and armJ GPQA
#      cells' own server.log -- not guessed.
#   3. The armJ multipl_e_100 re-run block is REMOVED. suite_gepo1.sh created
#      qwenhybridp24_q6k_b604 (0.8767, verified present). Re-running it would overwrite a
#      banked cell for no gain. Results are sacred.
#
# BASIS: not re-chosen. Every per_slot/parallel pair was read out of the comparator cell's
# OWN server.log. Every comparator cell records sampler=recommended (verified in preflight,
# all 10, before anything launches).
#
# COHORT = `recommended` (temp 0.6 / top_p 0.95 / top_k 20 / do_sample true), NOT greedy.
# This is a DIFFERENT cohort from the ream_arms/* cells run on 2026-08-26 (MPE-100 0.7533,
# LCB-48k 0.8312, both greedy/template_default). NEVER pool the two. Read
# summary.json.sampler.name before tabulating anything.
#
# NO LLAMA_EXTRA. This chain serves the GGUF's EMBEDDED template; the ream_arms chain passes
# a fixed jinja file. Carrying that flag over would silently change the prompt basis.
#
# SAMPLED => --use_cache IS A NO-OP (do_sample=true). Not resumable; a death restarts the
# current cell from 0. Inherent to the cohort's sampler, not a missing flag. Completed cells
# ARE skipped (summary.json check), so a restart of the SCRIPT resumes at cell granularity.
#
# WALL CLOCK: gepo1's identical 10 benches took 5.28h (gpqa 1.50 + lcb 1.51 + aime 0.93 +
# ifeval 0.42 + math500 0.35 + gsm8k 0.31 + the four short ones 0.26). gepo3 emits 22-29%
# fewer tokens on the code benches, so expect <= that, but do not treat a faster finish as
# evidence of anything -- it is the objective working.
#
# GPU1 ONLY. GPU0 is NOT ours.
set -uo pipefail

export CUDA_VISIBLE_DEVICES=1
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
export HF_HUB_ENABLE_HF_TRANSFER=0
export LM_EVAL_BIN=/root/anaconda3/envs/omnimergekit/bin/lm-eval
unset LLAMA_EXTRA            # embedded template -- see header
unset LLAMA_ARG_SPEC_TYPE    # no speculative decoding, same as the anchors

OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
RES=/srv/ml/eval_results/qwen_suite
LOG=/mnt/sdc/ml/brevity/gepo/suite_gepo3.log

GGUF=/mnt/sdc/ml/brevity/gepo/gguf_gepo3/a3b-gepo3-Q6_K.gguf
TOK=/mnt/sdc/ml/brevity/gepo/a3b-gepo3
NAME=qwena3bgepo3_q6k

REF=qwenhybridp24_q6k
PROFILE=qwen3_6
SAMPLER=recommended
PORT=8099
WORK_TBL=/mnt/sdc/ml/brevity/gepo

exec >>"$LOG" 2>&1
ts(){ date -u '+%F %T UTC'; }
say(){ echo ">>> [$(ts)] $*"; }
echo "=== suite_gepo3 start $(ts) ==="

# bench | per_slot | parallel   -- read back from the comparator cells' own server.log
JOBS=(
  "gpqa_diamond_full|45056|2"
  "gsm8k_100_boxed|45056|2"
  "arc_challenge_100|24576|2"
  "humaneval_full_think|24576|2"
  "humanevalplus_full_think|24576|2"
  "multipl_e_100|24576|2"
  "ifeval_100|24576|2"
  "math500_100_qwen|45056|2"
  "aime_30_qwen|69632|2"
  "lcb_v6_77q|45056|2"
)

# armJ's multipl_e_100 comparator is the post-bug-604 RE-RUN cell, not the pre-fix one.
comparator(){ [ "$1" = "multipl_e_100" ] && echo "${REF}_b604" || echo "$REF"; }

# ---------------- PRE-FLIGHT ----------------
FAIL=0
for p in "$GGUF" "$TOK/tokenizer.json" "$OMK/eval/omk_eval.py" "$OMK/eval/models/$PROFILE.yaml"; do
  [ -s "$p" ] || { say "MISSING: $p"; FAIL=1; }
done
[ "$(head -c4 "$GGUF")" = "GGUF" ] || { say "not a GGUF: $GGUF"; FAIL=1; }

# The GGUF must carry an imatrix, same as both arms it will be tabulated against.
"$OMKPY" - "$GGUF" <<'PY' || FAIL=1
import sys
from gguf import GGUFReader
kv = [k for k in GGUFReader(sys.argv[1]).fields if k.startswith("quantize.imatrix")]
print("[imat-gate] " + ("OK %d KV" % len(kv) if kv else "IMATRIX MISSING"))
raise SystemExit(0 if kv else 1)
PY

say "preflight: ${#JOBS[@]} benches -- template present + comparator on $SAMPLER"
for j in "${JOBS[@]}"; do
  B=${j%%|*}
  C=$(comparator "$B")
  [ -s "$OMK/eval/templates/$B.yaml" ] || { say "  MISSING template $B.yaml"; FAIL=1; }
  got=$("$OMKPY" - "$RES/$B/$C/summary.json" <<'PY' 2>/dev/null
import json,sys
try: print(((json.load(open(sys.argv[1])).get("sampler") or {}).get("name")) or "NONE")
except Exception: print("UNREADABLE")
PY
)
  if [ "$got" != "$SAMPLER" ]; then say "  PREFLIGHT_BAD $B comparator=$C sampler=$got want=$SAMPLER"; FAIL=1
  else say "  ok $B comparator=$C sampler=$got"; fi
done

free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
say "GPU1 free=${free}MiB"
[ "$free" -ge 60000 ] || { say "GPU1 only ${free}MiB free"; FAIL=1; }
rootfree=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
say "root fs free=${rootfree}G (floor 200G)"
[ "$rootfree" -ge 200 ] || { say "ROOT FS BELOW 200G FLOOR"; FAIL=1; }

[ "$FAIL" = 0 ] || { say "PREFLIGHT FAILED -- nothing launched"; exit 2; }
say "PREFLIGHT_OK"

# ---------------- RUN ----------------
run_cell(){   # served_name gguf tok bench slot par
  local nm=$1 gg=$2 tk=$3 B=$4 SLOT=$5 PAR=$6
  local TOTAL=$(( SLOT * PAR ))
  local s="$RES/$B/$nm/summary.json"
  if [ -f "$s" ]; then say "SKIP $B/$nm (summary exists)"; return 0; fi
  say "===== $B  $nm  per_slot=$SLOT par=$PAR total=$TOTAL sampler=$SAMPLER (NOT resumable)"
  local t0=$SECONDS
  "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$B" --quant q6_k \
      --model "$gg" --tokenizer "$tk" --served-name "$nm" --port "$PORT" \
      --results-dir "$RES" --parallel "$PAR" \
      --sampler-profile "$PROFILE" --sampler "$SAMPLER" \
      --metadata backend_args.llama_ctx=$TOTAL
  say "<<<< END $B/$nm rc=$? in $(( (SECONDS-t0)/60 ))m"

  # GATE 1: geometry -- per-slot ctx below the template's ask silently collapses slots
  local L="$RES/$B/$nm/server.log"
  local got sl
  got=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  sl=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  say "GEOMETRY $B/$nm per_slot=${got:-unknown} slots=${sl:-unknown} (want $SLOT / $PAR)"
  [ "${got:-0}" = "$SLOT" ] && [ "${sl:-0}" = "$PAR" ] \
      || say "!!! GEOMETRY MISMATCH $B/$nm -- NOT comparable to $REF"

  [ -f "$s" ] || { say "FAIL $B/$nm: no summary.json"; return 1; }

  # GATE 2: sampler provenance -- a greedy row must never enter this cohort's table
  local sn
  sn=$("$OMKPY" - "$s" <<'PY'
import json,sys
print(((json.load(open(sys.argv[1])).get("sampler") or {}).get("name")) or "NONE")
PY
)
  say "SAMPLER $B/$nm recorded=$sn (want $SAMPLER)"
  [ "$sn" = "$SAMPLER" ] || say "!!! SAMPLER MISMATCH $B/$nm -- NOT tableable against $REF"
  say "SCORE $B/$nm = $("$OMKPY" -c "
import json,sys
d=json.load(open('$s')); print(d.get('score'), d.get('metric'), d.get('filter'))
")"
  return 0
}

ran=0; failed=0
for j in "${JOBS[@]}"; do
  B=${j%%|*}; r=${j#*|}; SLOT=${r%%|*}; PAR=${r##*|}
  if run_cell "$NAME" "$GGUF" "$TOK" "$B" "$SLOT" "$PAR"; then ran=$((ran+1)); else failed=$((failed+1)); fi
done
say "=== gepo3 legs done ran=$ran failed=$failed ==="

# ---------------- TABLE (three arms, one cohort) ----------------
say "=== armJ vs gepo1 vs gepo2 vs gepo3 -- qwen_suite cohort (sampler=recommended) ==="
# Table lives in table_gepo_arms.py, not in a heredoc: a fourth arm had to be added
# and editing embedded Python inside a shell script is how basis errors get in.
"$OMKPY" "$WORK_TBL/table_gepo_arms.py"

echo "###### SUITE_GEPO3_DONE $(ts) ######"
