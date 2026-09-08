#!/usr/bin/env bash
# Canonical re-run of the two SUSPECT code benches for the three CoderX-release arms.
#
# WHY THIS EXISTS (user correction, 2026-08-21):
#   "Why you didn't use the omk canonical templates? They are done exactly to avoid messing
#    with different context sizes."
# Correct. The banked cells were served on hand-rolled geometry, two different ways:
#   (a) BOTH MPE runners (mpe_rebasis.sh, rerun_mpe_b604.sh) passed
#       `--metadata backend_args.llama_ctx=<override>`, discarding the template's 16384;
#   (b) humaneval_full_think.yaml pinned NO llama_ctx at all, so gpu_planner chose it from
#       free VRAM -- which is how two CoderX HE+ cells ended up served at 16384 vs 24576 and
#       became incomparable.
# Both templates now pin ctx AND parallel, and NOTHING here overrides them: no --metadata,
# and --parallel is passed only to match what the template already forces, so a mismatch is
# a loud failure instead of a silent re-plan.
#
#   multipl_e_100        16384 / 4 =  4096 per slot   (max_gen_toks  1024)
#   humaneval_full_think 49152 / 2 = 24576 per slot   (max_gen_toks 16384, think 12288)
#
# NEW RESULTS ROOT. Results are SACRED: nothing under /srv/ml/eval_results{,_b604} is touched
# or reused. A virgin cache also matters for MPE specifically -- gen_one() short-circuits on
# a sqlite cache AND a per-problem JSON that both store POST-extraction output, so reusing
# either would replay old completions and the geometry change would look like a no-op.
#
# THE LCB COLUMN IS NOT RE-RUN, AND THAT IS CORRECT. lcb_v6_77q_48k.yaml pins
# llama_ctx: 524288 and the cells ran at --parallel 8 => 65536/slot, template-pinned and
# matched across all three arms. It is already canonical; re-running it would only burn GPU.
#
# GATES (a cell that trips one is ABORTED, not scored):
#   GATE-0  template pins intact on THIS host (ctx, parallel, greedy) -- checked before launch
#   GATE-I  the bug-604 MultiPL-E extractor fix is INSTALLED here (md5 + golds), not just built
#   GATE-1  geometry READBACK from server.log == the template pin (bug-597)
#   GATE-2  sampler provenance: summary.json.sampler.name == template_default (greedy)
#   GATE-3  no ctx override recorded -- WEAK BY CONSTRUCTION, read this honestly: summary.json
#           does not persist resolved backend_args yet (open item #840), so this check has
#           nothing to inspect and passes vacuously today. GATE-1 is the one that actually
#           witnesses an override, because any override changes the served per-slot ctx.
#           Kept so it starts biting the day #840 lands; not counted as coverage until then.
#
# GPU PINNING. `--gpus` takes gpu IDs (auto|free|<ids csv>), NOT a count -- and with
# CUDA_VISIBLE_DEVICES=1 exported the visible device renumbers to 0, so `--gpus 1` would
# name a device that does not exist in the masked view. The export alone is the pattern the
# b604 runner used successfully on these exact cells: physical GPU1 becomes the only visible
# device, so the planner cannot reach GPU0 (which is NOT ours) even if it wanted to.
set -u
export CUDA_VISIBLE_DEVICES=1          # bs2 GPU1 is ours. GPU0 is NOT.
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
export HF_ALLOW_CODE_EVAL=1

OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
TDIR=$OMK/eval/templates
WORK=/mnt/sdc/ream-work
RES=/srv/ml/eval_results_canon
PORT=8099
TOK=/srv/ml/models/Qwen3.6-35B-A3B
GEN=$OMK/eval/multipl_e/multipl_e_generate.py
WANT_MD5=8a8e32c2728130ece233071cb858f6b2      # bug-604 fix + bug-607 raw retention
LOG=$WORK/canon_code.log

say(){ echo "[canon $(date -u +%H:%M:%S)Z] $*" | tee -a "$LOG"; }

# model cells: name | gguf
ARMS=(
  "coderx_armJ|$WORK/gguf/armJ_imat/armJ-Q6_K.gguf"
  "coder184e_pub|/srv/ml/models/gguf/Qwen3.6-35B-A3B-184e-coder-lcbmpe-GGUF/Qwen3.6-35B-A3B-184e-coder-lcbmpe-Q6_K.gguf"
  "base256e|/srv/ml/models/gguf/Qwen3.6-35B-A3B-256e-GGUF/Qwen3.6-35B-A3B-Q6_K.gguf"
)
# bench | expected per-slot | expected slots
BENCHES=(
  "multipl_e_100|4096|4"
  "humaneval_full_think|24576|2"
)

# ---------------- GATE-0: the templates on THIS host still pin what we claim ----------------
"$OMKPY" - "$TDIR" <<'PY' || exit 3
import sys, yaml
want = {"multipl_e_100": (16384, 4, 4096, 1024),
        "humaneval_full_think": (49152, 2, 24576, 16384)}
bad = 0
for name, (ctx, par, slot, mgt) in want.items():
    d = yaml.safe_load(open(f"{sys.argv[1]}/{name}.yaml"))
    ba, g = d["backend_args"], d["generation"]
    for label, got, exp in (("llama_ctx", ba.get("llama_ctx"), ctx),
                            ("llama_parallel", ba.get("llama_parallel"), par),
                            ("max_gen_toks", g.get("max_gen_toks"), mgt),
                            ("temperature", g.get("temperature"), 0.0),
                            ("do_sample", g.get("do_sample"), False)):
        if got != exp:
            print(f"GATE-0 FAIL {name}.{label}: {got!r} != {exp!r}"); bad = 1
    if ba.get("llama_ctx", 0) // max(1, ba.get("llama_parallel", 1)) != slot:
        print(f"GATE-0 FAIL {name}: per-slot != {slot}"); bad = 1
    print(f"GATE-0 {name}: ctx={ba.get('llama_ctx')} parallel={ba.get('llama_parallel')} "
          f"-> {slot}/slot, greedy, max_gen_toks={g.get('max_gen_toks')}")
sys.exit(bad)
PY
say "GATE-0 ok: both templates pin a complete geometry and are still greedy"

# ---------------- GATE-I: the bug-604 fix is INSTALLED, not merely built ----------------
[ -s "$GEN" ] || { say "REFUSE: $GEN missing"; exit 1; }
got=$(md5sum "$GEN" | awk '{print $1}')
[ "$got" = "$WANT_MD5" ] || { say "REFUSE GATE-I: $GEN md5=$got want=$WANT_MD5"; exit 3; }
"$OMKPY" "$GEN" --selftest >"$WORK/canon_selftest.log" 2>&1
grep -q "SELFTEST OK" "$WORK/canon_selftest.log" \
  || { say "REFUSE GATE-I: golds failed"; cat "$WORK/canon_selftest.log"; exit 3; }
say "GATE-I ok: extractor fix installed (md5 $got), $(grep -o 'SELFTEST OK ([0-9]*/[0-9]*)' "$WORK/canon_selftest.log")"

# ---------------- preflight: weights, disk, GPU ----------------
for a in "${ARMS[@]}"; do
  IFS='|' read -r cell g <<<"$a"
  [ -s "$g" ] || { say "REFUSE preflight: $cell gguf missing: $g"; exit 1; }
done
freeg=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "${freeg:-0}" -ge 210 ] || { say "REFUSE: root fs only ${freeg}G free (200G floor)"; exit 1; }
gfree=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
[ "${gfree:-0}" -gt 80000 ] || { say "REFUSE: GPU1 only ${gfree}MiB free"; exit 2; }
say "preflight ok: 3 GGUFs present, root ${freeg}G, GPU1 ${gfree}MiB free"

ran=0; aborted=0; failed=0; skipped=0
for b in "${BENCHES[@]}"; do
  IFS='|' read -r B SLOT PAR <<<"$b"
  for a in "${ARMS[@]}"; do
    IFS='|' read -r CELL G <<<"$a"
    out="$RES/$B/$CELL"

    if [ -f "$out/summary.json" ]; then say "SKIP $B/$CELL (already done)"; skipped=$((skipped+1)); continue; fi
    # a partial cell has a half-filled cache that is indistinguishable from a full one
    [ -d "$out" ] && mv "$out" "${out}_PARTIAL_$(date -u +%Y%m%dT%H%M%SZ)" \
      && say "preserved partial $B/$CELL"

    say "===== $B / $CELL  (template geometry: ${SLOT}/slot x ${PAR}) gguf=$(basename "$G")"
    # NO --metadata. --parallel only restates the template FORCE so a divergence is loud.
    "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$B" --quant q6_k \
        --model "$G" --tokenizer "$TOK" --served-name "$CELL" --port "$PORT" \
        --results-dir "$RES" --parallel "$PAR" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    say "<<<< END $B/$CELL rc=$rc"

    # ---- GATE-1: geometry readback (the whole point of this re-run) ----
    L="$out/server.log"
    gslot=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
    gsl=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
    say "GEOMETRY $B/$CELL per_slot=${gslot:-unknown} slots=${gsl:-unknown} (want $SLOT / $PAR)"
    if [ "${gslot:-0}" != "$SLOT" ] || [ "${gsl:-0}" != "$PAR" ]; then
      say "CANON_ABORT $B/$CELL: served geometry != template pin"; aborted=$((aborted+1)); continue
    fi

    S="$out/summary.json"
    [ -f "$S" ] || { say "FAIL $B/$CELL: no summary.json"; failed=$((failed+1)); continue; }

    # ---- GATE-2 (sampler) + GATE-3 (no ctx override) + the score ----
    "$OMKPY" - "$S" "$B/$CELL" "$SLOT" "$PAR" <<'PY'
import json, sys
s, tag, slot, par = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
d = json.load(open(s))
name = ((d.get("sampler") or {}).get("name")) or "NONE"
print(f"SAMPLER {tag}: recorded={name} (want template_default)")
if name != "template_default":
    print(f"CANON_ABORT {tag}: sampler is not the frozen greedy default"); sys.exit(9)
# GATE-3: nothing in the recorded metadata may carry a hand-set ctx. VACUOUS TODAY --
# summary.json does not persist backend_args (#840), so there is nothing here to catch an
# override; GATE-1's geometry readback is what actually proves the template pin was served.
meta_src = d.get("metadata") or d.get("overrides") or d.get("backend_args") or {}
if not meta_src:
    print(f"GATE-3 {tag}: VACUOUS (summary.json persists no backend_args -- see #840); "
          f"override coverage comes from GATE-1 only")
elif "llama_ctx" in json.dumps(meta_src):
    print(f"CANON_ABORT {tag}: a ctx override reached the run: {meta_src}"); sys.exit(9)
print(f"SCORE {tag}: {d.get('score')}  metric={d.get('metric')} filter={d.get('filter')} "
      f"n={d.get('n')} geometry={slot}/slot x {par}")
PY
    grc=$?
    if [ "$grc" = 9 ]; then aborted=$((aborted+1)); continue; fi
    ran=$((ran+1))
  done
done

say "=== CANON_CODE_DONE ran=$ran aborted=$aborted failed=$failed skipped=$skipped ==="
say "Root: $RES  (eval_results / eval_results_b604 untouched)"
say "LCB-v6-77q@48k is NOT here on purpose: it was already template-pinned (524288/8 ="
say "65536 per slot) and matched across all three arms, so it is already canonical."
echo "CANON_CODE_SENTINEL_DONE" | tee -a "$LOG"
