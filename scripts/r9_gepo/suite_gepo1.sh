#!/usr/bin/env bash
# Full omk canonical suite for a3b-gepo1, dropped into the EXISTING qwen_suite cohort
# alongside the banked armJ (qwenhybridp24_q6k) rows.
#
# BASIS: not re-chosen. Every per-slot/parallel pair below was read back out of the
# comparator cell's OWN server.log (`new slot, n_ctx =` / `n_slots =`), and every cell
# records sampler=recommended. That is the same reconstruction gate9c_armJ.sh applied,
# extended to the 4 benches gate9c did not cover (gate9d gapfill + gate9e arc).
#
# COHORT = `recommended` (temp 0.6 / top_p 0.95 / top_k 20 / do_sample true), NOT greedy.
# eval/models/qwen3_6.yaml marks greedy explicitly non-viable for this family on open-ended
# thinking benches. Never pool these rows with the ream_arms/* greedy cohort.
#
# NO LLAMA_EXTRA. gate9c serves the GGUF's EMBEDDED template; the ream_arms chain passes a
# fixed jinja. Carrying that flag over would silently change the prompt basis.
#
# SAMPLED => --use_cache IS A NO-OP (do_sample=true). Not resumable; a death restarts the
# cell from 0. Inherent to the cohort's sampler, not a missing flag.
#
# ---------------------------------------------------------------------------------------
# SCORER PROVENANCE (checked BEFORE launch, 2026-08-24) -- which armJ rows are comparable:
#   bs2's omk working copy took the three 2026-08-20 scorer fixes at 18:00:26 UTC
#   (multipl_e_generate.py / lcb_helpers.py mtimes). armJ cell finish times vs that:
#     gpqa 12:53, gsm8k 13:19, ifeval 13:45, humaneval 13:49, multipl_e 13:54,
#     lcb 15:24, humanevalplus 15:35, math500 16:14, aime 18:01, arc 19:10
#   - LCB (bug-606): armJ ran pre-fix, BUT the cell carries its own offline re-score
#     lcb_result.b606.json -> delta_pp 0.0, 64/64 n_pass, 75/75 controls reproduced,
#     0 fixed / 0 regressed, verdict OK. Proven no-op HERE. armJ LCB row is VALID.
#   - MPE (bug-604 chat_to_body): armJ ran pre-fix and CANNOT be repaired offline --
#     extraction happens at generation time and the cache predates bug-607 (measured
#     raw on 0/300 rows). So the banked armJ multipl_e_100 cell is VOID against a fresh
#     gepo1 run. It is RE-RUN at the end of this chain into a NEW cell (_b604); the old
#     cell is preserved untouched. Results are sacred.
#   - Everything else: unaffected by those two fixes (different scorers).
#   - Templates: humaneval_full_think.yaml has NO content diff (mtime touch only);
#     multipl_e_100.yaml only gained `llama_parallel: 4`, a HINT that the explicit CLI
#     --parallel overrides. No basis change.
# ---------------------------------------------------------------------------------------
#
# GPU1 ONLY. GPU0 is NOT ours.
set -uo pipefail

export CUDA_VISIBLE_DEVICES=1
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
export HF_HUB_ENABLE_HF_TRANSFER=0
export LM_EVAL_BIN=/root/anaconda3/envs/omnimergekit/bin/lm-eval
unset LLAMA_EXTRA            # embedded template -- see header
unset LLAMA_ARG_SPEC_TYPE    # no speculative decoding, same as gate9c

OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
RES=/srv/ml/eval_results/qwen_suite
LOG=/mnt/sdc/ml/brevity/gepo/suite_gepo1.log

GGUF=/mnt/sdc/ml/brevity/gepo/gguf_gepo1/a3b-gepo1-Q6_K.gguf
TOK=/mnt/sdc/ml/brevity/gepo/a3b-gepo1
NAME=qwena3bgepo1_q6k

AGGUF=/mnt/sdc/ream-work/gguf_armJ_hf/Qwen3.6-27B-A3B-CoderX-Q6_K.gguf
ATOK=/mnt/sdc/ream-work/armJ
REF=qwenhybridp24_q6k
AWANT_SHA=92bfdc9dca32f2ad81a85c6ba05d239cad6430fde112901245fb8b28f2ffa076

PROFILE=qwen3_6
SAMPLER=recommended
PORT=8099

exec >>"$LOG" 2>&1
ts(){ date -u '+%F %T UTC'; }
say(){ echo ">>> [$(ts)] $*"; }
echo "=== suite_gepo1 start $(ts) ==="

# bench | per_slot | parallel   -- read back from the REF cell's own server.log
# gpqa_diamond_full is ALREADY DONE for gepo1 (2026-08-24) and is deliberately absent.
JOBS=(
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

# ---------------- PRE-FLIGHT ----------------
FAIL=0
for p in "$GGUF" "$TOK/tokenizer.json" "$OMK/eval/omk_eval.py" "$OMK/eval/models/$PROFILE.yaml"; do
  [ -s "$p" ] || { say "MISSING: $p"; FAIL=1; }
done
[ "$(head -c4 "$GGUF")" = "GGUF" ] || { say "not a GGUF: $GGUF"; FAIL=1; }

say "preflight: ${#JOBS[@]} benches -- template present + comparator on $SAMPLER"
for j in "${JOBS[@]}"; do
  B=${j%%|*}
  [ -s "$OMK/eval/templates/$B.yaml" ] || { say "  MISSING template $B.yaml"; FAIL=1; }
  got=$("$OMKPY" - "$RES/$B/$REF/summary.json" <<'PY' 2>/dev/null
import json,sys
try: print(((json.load(open(sys.argv[1])).get("sampler") or {}).get("name")) or "NONE")
except Exception: print("UNREADABLE")
PY
)
  if [ "$got" != "$SAMPLER" ]; then say "  PREFLIGHT_BAD $B comparator sampler=$got want=$SAMPLER"; FAIL=1
  else say "  ok $B comparator sampler=$got"; fi
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

  # GATE 1: geometry
  local L="$RES/$B/$nm/server.log"
  local got sl
  got=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  sl=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  say "GEOMETRY $B/$nm per_slot=${got:-unknown} slots=${sl:-unknown} (want $SLOT / $PAR)"
  [ "${got:-0}" = "$SLOT" ] && [ "${sl:-0}" = "$PAR" ] \
      || say "!!! GEOMETRY MISMATCH $B/$nm -- NOT comparable to $REF"

  [ -f "$s" ] || { say "FAIL $B/$nm: no summary.json"; return 1; }

  # GATE 2: sampler provenance
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
say "=== gepo1 legs done ran=$ran failed=$failed ==="

# ---------------- armJ MPE re-run on the CURRENT extractor ----------------
# The banked armJ multipl_e_100 cell is pre-bug-604 and unrepairable offline (raw 0/300).
# New cell, old one preserved. Gated on the retained armJ receipt sha.
say "=== armJ multipl_e_100 re-run on the post-b604 extractor ==="
if [ -s "$AGGUF" ]; then
  GOT=$(sha256sum "$AGGUF" | cut -d' ' -f1)
  if [ "$GOT" = "$AWANT_SHA" ]; then
    say "SHA_OK armJ Q6_K bit-identical to the retained receipt"
    run_cell "${REF}_b604" "$AGGUF" "$ATOK" multipl_e_100 24576 2 || say "armJ MPE re-run FAILED"
  else
    say "SHA MISMATCH on $AGGUF -- got=$GOT want=$AWANT_SHA -- SKIPPING armJ MPE re-run"
  fi
else
  say "armJ Q6_K absent ($AGGUF) -- SKIPPING armJ MPE re-run"
fi

# ---------------- TABLE ----------------
say "=== a3b-gepo1 vs armJ -- qwen_suite cohort (sampler=recommended) ==="
"$OMKPY" - <<'PY'
import json, os
R = "/srv/ml/eval_results/qwen_suite"
NEW, REF = "qwena3bgepo1_q6k", "qwenhybridp24_q6k"
BENCH = ["gpqa_diamond_full", "gsm8k_100_boxed", "arc_challenge_100",
         "humaneval_full_think", "humanevalplus_full_think", "multipl_e_100",
         "ifeval_100", "math500_100_qwen", "aime_30_qwen", "lcb_v6_77q"]
# armJ's multipl_e_100 comparator is the RE-RUN cell, not the pre-b604 one.
OVERRIDE = {"multipl_e_100": REF + "_b604"}


def load(b, cell):
    try:
        return json.load(open(os.path.join(R, b, cell, "summary.json")))
    except Exception:
        return None


hdr = "%-26s %9s %9s %9s   %9s %9s %8s %7s" % (
    "bench", "armJ", "gepo1", "delta pp", "armJ tok", "gepo1 tok", "tok %", "sampler")
print(hdr); print("-" * len(hdr))
da = db = n = 0
for b in BENCH:
    a = load(b, OVERRIDE.get(b, REF))
    x = load(b, NEW)
    if not a or not x:
        print("%-26s %9s %9s" % (b, "-" if not a else round(a.get("score"), 4),
                                 "-" if not x else round(x.get("score"), 4)))
        continue
    ca = ((a.get("token_stats") or {}).get("completion_tokens") or {})
    cb = ((x.get("token_stats") or {}).get("completion_tokens") or {})
    ta, tb = ca.get("sum") or 0, cb.get("sum") or 0
    sa, sb = a.get("score"), x.get("score")
    smp = (x.get("sampler") or {}).get("name")
    tokpct = ("%+.1f%%" % ((tb - ta) / ta * 100)) if ta else "-"
    print("%-26s %9.4f %9.4f %+9.2f   %9d %9d %8s %7s"
          % (b, sa, sb, (sb - sa) * 100, ta, tb, tokpct, smp))
    da += sa; db += sb; n += 1
if n:
    print("-" * len(hdr))
    print("%-26s %9.4f %9.4f %+9.2f   (mean over %d benches)" % ("MEAN", da / n, db / n, (db - da) / n * 100, n))
print()
print("NOTE  multipl_e_100 armJ column = the _b604 RE-RUN cell (post-bug-604 extractor).")
print("      The pre-b604 armJ cell is preserved but VOID against a fresh run.")
print("      Sampled cohort (temp 0.6): every cell is ONE draw. Small per-bench deltas")
print("      are within sampling noise; only a repeat draw gives the band.")
PY

echo "###### SUITE_GEPO1_DONE $(ts) ######"
