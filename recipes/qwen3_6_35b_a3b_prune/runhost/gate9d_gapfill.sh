#!/usr/bin/env bash
# CoderX gate 1b (v2) — REPAIR + GAP-FILL, BOTH ARMS, on the published sampler basis.
#
# WHY BOTH ARMS ON ALL FOUR. None of these four benches has a usable comparator cell:
#   humanevalplus_full_think  the published cut's cell was SIGINT-killed, no summary.json
#   arc_challenge_100         brand-new template, no cell has ever existed
#   math500_100_qwen          replaces a VOID cell (stop-word collision, see below)
#   aime_30_qwen              replaces a bench that was about to hit the same collision
# so each costs a full run per arm. 8 cells total.
#
# ============================ THE STOP-WORD COLLISION ============================
# minerva_math500 resolves `until: ["Problem:"]`; aime24_chat and arc_challenge_full_chat
# resolve `until: ["Question:", ...]`. All three are num_fewshot: 0 chat tasks, so the
# delimiter is VESTIGIAL -- there is no next exemplar for it to delimit. But llama.cpp
# matches stop words on the RAW token stream, BEFORE `--reasoning-format deepseek` splits
# <think> into reasoning_content. Qwen3.x opens nearly every answer with an enumerated plan
# whose first item is "**Understand the Problem:**" / "**Understand the Question:**", so the
# stop fires INSIDE the thinking block and terminates the entire generation ~65 chars in.
#
# MEASURED 2026-08-20 on the math500_100 pair this cohort already produced:
#   armJ  reasoning p50=65 chars / content p50=0 / 80 of 100 rows under 100 chars -> 0.2200
#   pub   reasoning p50=0 (skips thinking on 63/100)                              -> 0.7000
#   0 of 200 completions across BOTH arms contain the delimiter anywhere. The 80 short armJ
#   rows collapse to 5 distinct prefixes, each breaking exactly where it would be written.
#   OMK_CAP_CHECK CLEAN (0/100 capped), 0 empty -> not truncation, not emptiness.
# A matched-but-defective basis is still invalid: it differentially penalises whichever arm
# thinks MORE, which is the axis under test. Both cells were parked *_STOPWORD_INVALID_<ts>.
#
# T177 (2026-06-01) found and fixed exactly this for gsm8k + math500 but never covered
# aime24_chat or arc_challenge_full_chat. The qwen-safe task variants now exist for all of
# them; this chain runs only those. NEW BASIS -- never pool with the old cells.
#
# GATE 0 is the generalisation of the fix: it refuses if ANY task about to run resolves an
# `until` containing a plain-text delimiter. Gate the taxonomy, not the last bug.
#
# GEOMETRY. aime_30_qwen pins 69632/slot because max_gen_toks is 65536 -- the old cohort ran
# aime_30 at 45056/slot, i.e. a slot SMALLER than the generation budget (SAT_COLLAPSE-shaped).
# llama_ctx is the TOTAL pool, divided by --parallel (bug-597), so total = per_slot * par and
# only the `new slot, n_ctx = N` readback proves it.
#
# ORDERING. Waits on GATE10_DONE: gate9c -> gate10 (publish-blocking loop gate) -> gate9d.
set -u
export CUDA_VISIBLE_DEVICES=1          # bs2 GPU1 is ours. GPU0 is NOT.
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
# GATE0 below LOADS the humaneval_plus_chat_think task, whose utils_chat.py calls
# code_eval.compute() at *import* time -- that refuses without this. omk_eval sets it for
# the run itself, but GATE0 runs in this shell, before omk_eval exists. (bug-594)
export HF_ALLOW_CODE_EVAL=1
command -v lm-eval >/dev/null || { echo "REFUSING: lm-eval not on PATH"; exit 1; }

OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
WORK=/mnt/sdc/ream-work
RES=/srv/ml/eval_results/qwen_suite
PORT=8099
SAMPLER=recommended
PROFILE=qwen3_6
WAIT_MAX_S=${WAIT_MAX_S:-64800}          # 18 h: gate9c + gate10 both have to clear
POLL_S=180

say(){ echo "[gate9d $(date -u +%H:%M:%S)Z] $*"; }

# arm | served-name | gguf | tokenizer
ARMS=(
  "pub|qwencodermpe_q6k|/srv/ml/models/gguf/Qwen3.6-35B-A3B-184e-coder-lcbmpe-GGUF/Qwen3.6-35B-A3B-184e-coder-lcbmpe-Q6_K.gguf|/srv/ml/models/Qwen3.6-35B-A3B"
  "armJ|qwenhybridp24_q6k|$WORK/gguf/armJ_imat/armJ-Q6_K.gguf|$WORK/armJ"
)
# bench | per_slot | parallel
JOBS=(
  "humanevalplus_full_think|24576|2"
  "arc_challenge_100|24576|2"
  "math500_100_qwen|45056|2"
  "aime_30_qwen|69632|2"
)

# ---- preflight: templates + weights ----
for j in "${JOBS[@]}"; do
  B=${j%%|*}
  [ -s "$OMK/eval/templates/$B.yaml" ] || { say "REFUSE: template $B.yaml not installed"; exit 1; }
done
for a in "${ARMS[@]}"; do
  IFS='|' read -r arm name g tok <<<"$a"
  [ -s "$g" ]   || { say "REFUSE: $arm gguf missing: $g"; exit 1; }
  [ -d "$tok" ] || { say "REFUSE: $arm tokenizer missing: $tok"; exit 1; }
  say "preflight ok: $arm -> $(basename "$g")"
done

# ---- GATE 0 (FATAL): no plain-text stop delimiter in any task we are about to run ----
say "GATE0: resolving until= for every task in JOBS"
"$OMKPY" - "$OMK/eval" "${JOBS[@]}" <<'PY' || { say "GATE9D_REFUSE: GATE0 stop-word check failed"; exit 3; }
import sys, yaml, os
from lm_eval.tasks import TaskManager
base = sys.argv[1]
jobs = [a.split("|")[0] for a in sys.argv[2:]]
BAD = ("Question:", "Problem:", "Answer:", "\n\n")
ok = True
for b in jobs:
    t = yaml.safe_load(open(os.path.join(base, "templates", b + ".yaml")))
    task = t["task"]
    inc = (t.get("backend_args") or {}).get("lm_eval_include_path")
    tm = TaskManager(include_path=os.path.join(base, inc)) if inc else TaskManager()
    if task not in tm.all_tasks:
        print("  %-26s task=%s NOT REGISTERED" % (b, task)); ok = False; continue
    gk = tm.load_task_or_group([task])[task].config.generation_kwargs or {}
    u = gk.get("until") or []
    hit = [s for s in u if s in BAD]
    print("  %-26s task=%-30s until=%s %s" % (b, task, u, ("<<< POISONED " + str(hit)) if hit else "ok"))
    if hit: ok = False
print("GATE0_OK" if ok else "GATE0_FAIL")
sys.exit(0 if ok else 1)
PY
say "GATE0 passed: no plain-text delimiter in any scheduled task"

# ---- readiness: gate10 finished AND GPU1 actually free ----
t0=$(date +%s)
while :; do
  d10=0; grep -aq "GATE10_DONE" "$WORK/gate10_loop.log" 2>/dev/null && d10=1
  srv=$(pgrep -c -f "llama-server" 2>/dev/null || true)
  free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || true)
  if [ "$d10" = 1 ] && [ "$srv" = 0 ] && [ "${free:-0}" -gt 80000 ]; then
    say "READY: gate10 done, no server alive, GPU1 free=${free}MiB"; break
  fi
  el=$(( $(date +%s) - t0 ))
  if [ "$el" -ge "$WAIT_MAX_S" ]; then
    say "REFUSE: not ready after ${el}s (gate10_done=$d10 server=$srv gpu1_free=${free}MiB)"; exit 2
  fi
  [ $(( el % 3600 )) -lt "$POLL_S" ] && say "waiting ${el}s (gate10_done=$d10 server=$srv gpu1_free=${free}MiB)"
  sleep "$POLL_S"
done

ran=0; failed=0; aborted=0
for j in "${JOBS[@]}"; do
  B=${j%%|*}; r=${j#*|}; SLOT=${r%%|*}; PAR=${r##*|}; TOTAL=$(( SLOT * PAR ))
  for a in "${ARMS[@]}"; do
    IFS='|' read -r arm NAME G TOK <<<"$a"
    cell="$RES/$B/$NAME"
    [ -f "$cell/summary.json" ] && { say "SKIP $B/$arm (scored cell exists)"; continue; }
    # A dir with no summary.json is an interrupted run. PRESERVE it, never delete -- its
    # sqlite cache makes the re-run resumable and its server.log is geometry evidence.
    if [ -d "$cell" ] && [ ! -d "$cell/sqlite_cache" ]; then
      mv "$cell" "${cell}_INTERRUPTED_$(date -u +%Y%m%dT%H%M%SZ)" && say "preserved interrupted $B/$arm"
    elif [ -d "$cell" ]; then
      say "REUSING $B/$arm partial cell (sqlite cache present -> resumable)"
    fi

    say "===== $B / $arm  per_slot=$SLOT par=$PAR total=$TOTAL sampler=$SAMPLER"
    "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$B" --quant q6_k \
        --model "$G" --tokenizer "$TOK" --served-name "$NAME" --port "$PORT" \
        --results-dir "$RES" --parallel "$PAR" \
        --sampler-profile "$PROFILE" --sampler "$SAMPLER" \
        --metadata backend_args.llama_ctx=$TOTAL
    say "<<<< END $B/$arm rc=$?"

    # ---- GATE 1: geometry readback ----
    L="$cell/server.log"
    got=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
    sl=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
    say "GEOMETRY $B/$arm per_slot=${got:-unknown} slots=${sl:-unknown} (want $SLOT / $PAR)"
    if [ "${got:-0}" != "$SLOT" ] || [ "${sl:-0}" != "$PAR" ]; then
      say "GATE9D_ABORT $B/$arm: geometry not honoured"; aborted=$((aborted+1)); continue; fi

    s="$cell/summary.json"
    [ -f "$s" ] || { say "FAIL $B/$arm: no summary.json"; failed=$((failed+1)); continue; }

    # ---- GATE 2: sampler provenance ----
    sn=$("$OMKPY" - "$s" <<'PY'
import json,sys
print(((json.load(open(sys.argv[1])).get("sampler") or {}).get("name")) or "NONE")
PY
)
    say "SAMPLER $B/$arm recorded=$sn (want $SAMPLER)"
    [ "$sn" = "$SAMPLER" ] || { say "GATE9D_ABORT $B/$arm: sampler mismatch"; aborted=$((aborted+1)); continue; }

    # ---- GATE 3 (NEW, reporting): short-completion census -- the collision's fingerprint ----
    "$OMKPY" - "$cell" "$B" "$arm" <<'PY'
import json,sys,glob,os
cell,b,arm=sys.argv[1],sys.argv[2],sys.argv[3]
f=sorted(glob.glob(os.path.join(cell,"lm_eval_out","*","samples_*.jsonl")))
if not f: print("   SHORTCENSUS %s/%s: no samples file" % (b,arm)); raise SystemExit
L=[json.loads(l) for l in open(f[-1]) if l.strip()]
def resp(d):
    r=d.get("resps") or d.get("filtered_resps") or []
    while isinstance(r,list) and r: r=r[0]
    return r if isinstance(r,str) else ""
R=[resp(d) for d in L]
n=len(R); short=sum(1 for x in R if len(x)<100); empty=sum(1 for x in R if not x.strip())
pref=len(set(x[:40] for x in R if len(x)<100))
flag=" <<< COLLISION-SHAPED" if n and short/n>0.30 and pref<=8 else ""
print("   SHORTCENSUS %s/%s: n=%d under100=%d empty=%d distinct_short_prefixes=%d%s"
      % (b,arm,n,short,empty,pref,flag))
PY

    ran=$((ran+1))
    say "SCORE $B/$arm = $("$OMKPY" - "$s" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); print(d.get("score"), d.get("metric"), d.get("filter"))
PY
)"
  done
done
say "=== GATE9D_DONE ran=$ran failed=$failed aborted=$aborted ==="
say "REMINDER: arc_challenge_100 is a guardrail (n=100, sigma ~2.6pp). Report it with its n,"
say "never beside a full-1172 ARC figure. math500_100_qwen and aime_30_qwen are a NEW BASIS:"
say "never pool them with the parked *_STOPWORD_INVALID_* cells or with any Gemma-4 cell."
