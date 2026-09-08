#!/usr/bin/env bash
# MPE-100 re-run after the bug-604 chat_to_body fix.
#
# WHY. multipl_e_generate.py:chat_to_body fell back to `code[code.find("{") + 1:]` when the
# model's reply did not restate the signature. For a BODY-ONLY reply that first '{' belongs to
# the first loop/if, so the fallback ate its opening line and orphaned the matching '}'. The
# class closed early, MultiPL-E's appended main() landed outside it, and javac reported
# "illegal start of type" -- a GUARANTEED zero banked as a model failure. Measured over the
# 9 qwen_suite arms: humaneval-java 176/900 eaten, 176 of 176 FATAL; rs 27 eaten / 25 fail;
# js 895 eaten but benign (tests_supply_close is False for js, negative balance is normal).
# Fixed + gold-tested 2026-08-20 (omk 3221739). The java column was measuring how often each
# arm restates the signature, not Java ability.
#
# ============================ THE CACHE TRAP -- READ THIS ============================
# gen_one() short-circuits on TWO caches BEFORE the model is ever called, and BOTH store the
# POST-transform completion:
#     1) sqlite  scache[key] = {"completion": completion, ...}   (multipl_e_generate.py:369)
#     2) the per-problem JSON  already_generated(out_path)       (multipl_e_generate.py:352)
# A re-run that reuses either would replay the exact broken bodies and score BYTE-IDENTICALLY
# -- the fix would look like a no-op. omk derives BOTH from the cell dir
# (cache_dir = out_dir/"sqlite_cache", gen_dir = out_dir/"generations"), so this script writes
# to a SEPARATE results root ($RES_NEW) and every cell gets a virgin cache and out-dir.
# That also means the OLD cells are never touched: results are SACRED, nothing is overwritten.
# ====================================================================================
#
# GATES
#   GATE-I  INSTALLED, not merely built: multipl_e_generate.py on THIS host must be the fixed
#           file (md5 == the pushed one) AND --selftest must pass (12/12 since bug-607). A built fix that is not
#           installed is not a fix.
#   GATE-1  geometry readback (bug-597: llama_ctx is the TOTAL pool, divided by --parallel).
#   GATE-2  sampler provenance -- summary.json.sampler.name must equal the cell's ORIGINAL
#           sampler, so a greedy cell can never be silently pooled with a sampled one.
#   GATE-B  bug-604 differential control: count java bodies with negative brace balance in the
#           OLD cell vs the NEW one. If the old cell had unbalanced bodies and the new cell has
#           just as many, the fix did NOT take effect on this run -> REFUSE the cell. This is
#           the only check that can tell "fix applied" from "fresh sample, same bug".
#
# ORDERING. Armed on GATE9E_DONE so it can never contend for GPU1:
#     gate9c -> gate9d -> gate10b -> gate9e -> THIS
# Tier-ordered so the decision-relevant CoderX cells (T1) land first; safe to stop at any
# tier boundary.
#
# BASIS NOTE (do not lose this). qwen_suite/* ran sampler=recommended (temp 0.6/top_p 0.95/
# top_k 20); ream_arms/* ran greedy (template_default). These are TWO COHORTS and must never
# be pooled. Each cell below is re-run on ITS OWN original sampler and geometry.
#
# UNRECORDED-AXIS NOTE. qwencodermpe_q6k and qwencodermpe_t10_q6k record the SAME weights, the
# SAME sampler block and the SAME geometry -- whatever "t10" varied was never recorded, so it
# is not provenance. Treat that pair as a same-basis REPEAT (0.82 vs 0.84 = a 2pp MPE-100
# jitter band), not as a temperature contrast.
set -u
export CUDA_VISIBLE_DEVICES=1          # bs2 GPU1 is ours. GPU0 is NOT.
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
export HF_ALLOW_CODE_EVAL=1

OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
WORK=/mnt/sdc/ream-work
RES_OLD=/srv/ml/eval_results
RES_NEW=/srv/ml/eval_results_b604
PORT=8099
B=multipl_e_100
TOK_BASE=/srv/ml/models/Qwen3.6-35B-A3B
GEN=$OMK/eval/multipl_e/multipl_e_generate.py
WANT_MD5=8a8e32c2728130ece233071cb858f6b2     # bug-604 fix + bug-607 raw retention; byte-identical on both hosts
WAIT_MAX_S=${WAIT_MAX_S:-86400}
POLL_S=180
TIERS=${TIERS:-1,2,3}                          # e.g. TIERS=1 to stop after the CoderX cells

say(){ echo "[b604 $(date -u +%H:%M:%S)Z] $*"; }

# tier | cohort | cell | gguf | sampler | per_slot | parallel
ARMS=(
  # ---- T1: decision-critical for the CoderX (R6) release -- pub vs armJ on BOTH cohorts ----
  "1|qwen_suite|qwencodermpe_q6k|/srv/ml/models/gguf/Qwen3.6-35B-A3B-184e-coder-lcbmpe-GGUF/Qwen3.6-35B-A3B-184e-coder-lcbmpe-Q6_K.gguf|recommended|24576|2"
  "1|qwen_suite|qwenhybridp24_q6k|$WORK/gguf/armJ_imat/armJ-Q6_K.gguf|recommended|24576|2"
  "1|ream_arms|pub184e_imat|/srv/ml/models/gguf/Qwen3.6-35B-A3B-184e-coder-lcbmpe-GGUF/Qwen3.6-35B-A3B-184e-coder-lcbmpe-Q6_K.gguf|template_default|12288|4"
  "1|ream_arms|hybrid_p24_ourssal_reapfloor_slot12288|$WORK/gguf/armJ_imat/armJ-Q6_K.gguf|template_default|12288|4"
  # ---- T2: remaining qwen_suite arms whose GGUF survived ----
  "2|qwen_suite|qwen256e_q6k|/srv/ml/models/gguf/Qwen3.6-35B-A3B-256e-GGUF/Qwen3.6-35B-A3B-Q6_K.gguf|recommended|24576|2"
  "2|qwen_suite|qwencodermpe_t10_q6k|/srv/ml/models/gguf/Qwen3.6-35B-A3B-184e-coder-lcbmpe-GGUF/Qwen3.6-35B-A3B-184e-coder-lcbmpe-Q6_K.gguf|recommended|24576|2"
  # ---- T3: the rest of the REAM arm matrix (all GGUFs present) ----
  "3|ream_arms|base256e_imat|/srv/ml/models/gguf/Qwen3.6-35B-A3B-256e-GGUF/Qwen3.6-35B-A3B-Q6_K.gguf|template_default|12288|4"
  "3|ream_arms|reamB_reapsal_merge|$WORK/gguf/armB_imat/armB-Q6_K.gguf|template_default|12288|4"
  "3|ream_arms|reamC_ourssal_merge|$WORK/gguf/armC_imat/armC-Q6_K.gguf|template_default|12288|4"
  "3|ream_arms|reamC_noimat_ctrl|$WORK/gguf/armC_noimat/armC-Q6_K.gguf|template_default|12288|4"
  "3|ream_arms|reamD_ourssal_nomerge|$WORK/gguf/armD_ourssal_nomerge_imat/armD_ourssal_nomerge-Q6_K.gguf|template_default|12288|4"
  "3|ream_arms|reamD_rpt|$WORK/gguf/armD_ourssal_nomerge_imat/armD_ourssal_nomerge-Q6_K.gguf|template_default|12288|4"
  "3|ream_arms|reamE_reapsal_nomerge|$WORK/gguf/armE_imat/armE-Q6_K.gguf|template_default|12288|4"
  "3|ream_arms|reamF_rnorm_nomerge|$WORK/gguf/armF_rnorm_nomerge_imat/armF_rnorm_nomerge-Q6_K.gguf|template_default|12288|4"
  "3|ream_arms|reamG_ourssal_merge_gs2|$WORK/gguf/armG_imat/armG-Q6_K.gguf|template_default|12288|4"
  "3|ream_arms|reamH_ourssal_merge_gs4|$WORK/gguf/armH_imat/armH-Q6_K.gguf|template_default|12288|4"
  "3|ream_arms|hybrid_p12_ourssal_reapfloor_slot12288|$WORK/gguf/armI_imat/armI-Q6_K.gguf|template_default|12288|4"
)
#
# NOT RE-RUNNABLE (5 of the 9 qwen_suite arms). Their Q6_K GGUFs are gone from disk, their
# bf16 sources are gone, and no -GGUF repo for them exists on HF (probe gold-checked against
# a known-present and a known-absent repo). They cannot be regenerated, so their java column
# stays poisoned and must be struck from any table rather than compared:
#     qwen184e_q6k     qwenadd16_q6k     qwencoder_q6k     qwensh12_q6k     qwensh12t10_q6k
#

# ---------------------- GATE-I: the fix is INSTALLED on this host ----------------------
[ -s "$GEN" ] || { say "REFUSE: $GEN missing"; exit 1; }
got=$(md5sum "$GEN" | awk '{print $1}')
[ "$got" = "$WANT_MD5" ] || { say "REFUSE GATE-I: $GEN md5=$got want=$WANT_MD5 (unfixed/stale copy)"; exit 3; }
"$OMKPY" "$GEN" --selftest >/tmp/b604_selftest.log 2>&1
grep -q "SELFTEST OK" /tmp/b604_selftest.log || { say "REFUSE GATE-I: golds failed"; cat /tmp/b604_selftest.log; exit 3; }
say "GATE-I ok: fix installed (md5 $got), golds $(grep -o 'SELFTEST OK ([0-9]*/[0-9]*)' /tmp/b604_selftest.log)"

# disk: bs2 root fs must keep >=200G free ALWAYS
freeg=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "${freeg:-0}" -ge 210 ] || { say "REFUSE: root fs only ${freeg}G free (floor 200G + margin)"; exit 1; }
say "disk ok: / has ${freeg}G free"

for a in "${ARMS[@]}"; do
  IFS='|' read -r t co cell g sm slot par <<<"$a"
  [ -s "$g" ] || { say "REFUSE preflight: $co/$cell gguf missing: $g"; exit 1; }
done
say "preflight ok: ${#ARMS[@]} cells, all GGUFs present"

# ---------------------- readiness: gate9e done AND GPU1 actually free ----------------------
# `pgrep -c` PRINTS 0 and EXITS 1 on no-match; `|| echo 0` would append a SECOND zero and the
# compare could never be true (bug-601). Use `|| true`.
t0=$(date +%s)
while :; do
  d=0; grep -aq "GATE9E_DONE" "$WORK/gate9e_arc.log" 2>/dev/null && d=1
  srv=$(pgrep -c -f "llama-server" 2>/dev/null || true); srv=${srv:-0}
  lme=$(pgrep -c -f "bin/lm-eval" 2>/dev/null || true); lme=${lme:-0}
  free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || true)
  if [ "$d" = 1 ] && [ "$srv" = 0 ] && [ "$lme" = 0 ] && [ "${free:-0}" -gt 80000 ]; then
    say "READY: gate9e done, nothing alive, GPU1 free=${free}MiB"; break
  fi
  el=$(( $(date +%s) - t0 ))
  if [ "$el" -ge "$WAIT_MAX_S" ]; then
    say "REFUSE: not ready after ${el}s (gate9e_done=$d server=$srv lm_eval=$lme gpu1_free=${free}MiB)"; exit 2
  fi
  [ $(( el % 3600 )) -lt "$POLL_S" ] && say "waiting ${el}s (gate9e_done=$d server=$srv lm_eval=$lme gpu1_free=${free}MiB)"
  sleep "$POLL_S"
done

ran=0; failed=0; aborted=0; skipped=0
for a in "${ARMS[@]}"; do
  IFS='|' read -r TIER CO CELL G SM SLOT PAR <<<"$a"
  case ",$TIERS," in *",$TIER,"*) ;; *) skipped=$((skipped+1)); continue;; esac

  newdir="$RES_NEW/$CO/$B/$CELL"
  olddir="$RES_OLD/$CO/$B/$CELL"
  [ -f "$newdir/summary.json" ] && { say "SKIP T$TIER $CO/$CELL (already re-run)"; continue; }
  # An interrupted cell has a poisoned-free but PARTIAL cache. Preserve it under a timestamp
  # rather than reuse it -- a half-filled cache is indistinguishable from a full one here.
  if [ -d "$newdir" ]; then
    mv "$newdir" "${newdir}_PARTIAL_$(date -u +%Y%m%dT%H%M%SZ)" && say "preserved partial $CO/$CELL"
  fi

  TOK="$TOK_BASE"
  for cand in "$WORK/$(echo "$CELL" | grep -oE 'arm[A-J]' || true)" "$WORK/${CELL%%_*}"; do
    [ -n "$cand" ] && [ -d "$cand" ] && TOK="$cand" && break
  done

  TOTAL=$(( SLOT * PAR ))
  say "===== T$TIER $CO/$CELL  per_slot=$SLOT par=$PAR total=$TOTAL sampler=$SM"
  say "      gguf=$(basename "$G")  tok=$TOK  out=$newdir (VIRGIN cache -- no poisoned replay)"

  if [ "$SM" = "template_default" ]; then
    "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$B" --quant q6_k \
        --model "$G" --tokenizer "$TOK" --served-name "$CELL" --port "$PORT" \
        --results-dir "$RES_NEW/$CO" --parallel "$PAR" \
        --metadata backend_args.llama_ctx=$TOTAL
  else
    "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$B" --quant q6_k \
        --model "$G" --tokenizer "$TOK" --served-name "$CELL" --port "$PORT" \
        --results-dir "$RES_NEW/$CO" --parallel "$PAR" \
        --sampler-profile qwen3_6 --sampler "$SM" \
        --metadata backend_args.llama_ctx=$TOTAL
  fi
  rc=$?
  say "<<<< END $CO/$CELL rc=$rc"

  # ---- GATE 1: geometry readback ----
  L="$newdir/server.log"
  gslot=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  gsl=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  say "GEOMETRY $CO/$CELL per_slot=${gslot:-unknown} slots=${gsl:-unknown} (want $SLOT / $PAR)"
  if [ "${gslot:-0}" != "$SLOT" ] || [ "${gsl:-0}" != "$PAR" ]; then
    say "B604_ABORT $CO/$CELL: geometry not honoured"; aborted=$((aborted+1)); continue; fi

  S="$newdir/summary.json"
  [ -f "$S" ] || { say "FAIL $CO/$CELL: no summary.json"; failed=$((failed+1)); continue; }

  # ---- GATE 2: sampler provenance ----
  sn=$("$OMKPY" - "$S" <<'PY'
import json,sys
print(((json.load(open(sys.argv[1])).get("sampler") or {}).get("name")) or "NONE")
PY
)
  say "SAMPLER $CO/$CELL recorded=$sn (want $SM)"
  [ "$sn" = "$SM" ] || { say "B604_ABORT $CO/$CELL: sampler mismatch"; aborted=$((aborted+1)); continue; }

  # ---- GATE B: the bug-604 differential control ----
  # A sampled re-run changes completions anyway, so "they differ" proves nothing. What IS
  # decisive is the defect itself: java bodies whose braces close more than they open cannot
  # compile. The old cell has N of them; the fixed cell must have far fewer.
  "$OMKPY" - "$olddir" "$newdir" "$CO/$CELL" <<'PY'
import json, sys, glob, os
old, new, tag = sys.argv[1], sys.argv[2], sys.argv[3]
def census(root):
    d = os.path.join(root, "generations", "humaneval-java")
    n = bad = 0; names = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try: j = json.load(open(f))
        except Exception: continue
        c = (j.get("completions") or [""])[0]
        n += 1
        if c.count("{") - c.count("}") < 0:
            bad += 1; names.append(j.get("name") or os.path.basename(f))
    return n, bad, names
no, bo, _ = census(old)
nn, bn, nb = census(new)
print(f"GATE-B {tag}: java unbalanced OLD {bo}/{no} -> NEW {bn}/{nn}")
if no == 0:
    print(f"GATE-B {tag}: NOTE no old java generations to compare (control unavailable)")
elif bo > 0 and bn >= bo:
    print(f"GATE-B {tag}: FAIL — defect did not shrink; the fix did not take on this run")
    print(f"GATE-B {tag}: still-unbalanced: {nb[:10]}")
    sys.exit(9)
else:
    print(f"GATE-B {tag}: PASS — {bo - bn} java cells recovered from guaranteed-zero")
PY
  brc=$?
  if [ "$brc" = 9 ]; then say "B604_ABORT $CO/$CELL: GATE-B failed"; aborted=$((aborted+1)); continue; fi

  # ---- score delta, per language (java is the affected column) ----
  "$OMKPY" - "$olddir" "$newdir" "$CO/$CELL" <<'PY'
import json, sys, os
old, new, tag = sys.argv[1], sys.argv[2], sys.argv[3]
def rd(root):
    s = os.path.join(root, "summary.json"); m = os.path.join(root, "mpe_result.json")
    sc = json.load(open(s)).get("score") if os.path.exists(s) else None
    pl = (json.load(open(m)).get("per_lang") or {}) if os.path.exists(m) else {}
    return sc, pl
so, po = rd(old); sn_, pn = rd(new)
def f(x): return "n/a" if x is None else f"{x:.4f}"
print(f"SCORE {tag}: overall {f(so)} -> {f(sn_)}")
for lang in ("java", "rs", "js"):
    a = (po.get(lang) or {}).get("pass_at_1"); b = (pn.get(lang) or {}).get("pass_at_1")
    if a is not None or b is not None:
        print(f"SCORE {tag}:   {lang:<5} {f(a)} -> {f(b)}")
PY
  ran=$((ran+1))
done

say "=== B604_RERUN_DONE ran=$ran failed=$failed aborted=$aborted skipped_tier=$skipped ==="
say "New cells live under $RES_NEW — the OLD cells are untouched and must be RETIRED, not"
say "compared: a pre-fix java column and a post-fix one are different measurements."
say "5 qwen_suite arms (qwen184e/qwenadd16/qwencoder/qwensh12/qwensh12t10) are UNRUNNABLE —"
say "GGUF gone from disk, bf16 gone, no HF repo. Strike their java column, do not compare it."
