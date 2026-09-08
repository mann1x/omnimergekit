#!/usr/bin/env bash
# CoderX gate 1c (gate9e) — the ONE cell gate9d could not run: arc_challenge_100, both arms.
#
# WHY THIS EXISTS. gate9d aborted arc_challenge_100 on BOTH arms at 15:35:50Z with rc=1,
# before serving anything:
#     arc_challenge_100.yaml: selection.type must be one of
#     ['explicit','filter','indices','langs','problems']
# That was MY template bug, not the model's: I wrote `selection.type: first_n`, which is not
# in template_loader.KNOWN_SELECTION_TYPES (eval/templates/template_loader.py:56). The
# geometry gate then saw `per_slot=unknown` and correctly refused to bank a null cell.
# Fixed 2026-08-20 to `type: filter / all: true` -- byte-identical to the parent
# arc_challenge_full.yaml, which is what this template's own header already promised. The
# first-100 selection has ALWAYS come from lm-eval `--limit`, derived from template.n; the
# selection block never drove it. Verified: template_loader.load("arc_challenge_100") now
# returns task=arc_challenge_full_chat_qwen n=100 selection={'type':'filter','all':True}.
#
# gate9d does NOT retry an aborted cell, so this bench needs its own runner. It is ordered
# AFTER gate10b (which is itself armed on GATE9D_DONE) so nothing ever contends for GPU1:
#     gate9c -> gate9d -> gate10b -> gate9e
#
# BASIS. Same as every other gate9d cell: sampler=recommended on profile qwen3_6, Q6_K
# imatrix GGUFs, results under qwen_suite/. This is a 100-question draw, NOT comparable to
# anyone's full-1172 ARC number -- report it with its n (see the template header).
#
# THE THREE GATES ARE gate9d's, UNCHANGED -- this is a gap-filler, not a new methodology:
#   GATE0  no plain-text `until` delimiter in the task about to run (bug-599 taxonomy).
#          arc_challenge_full_chat_qwen resolves until=['</s>','<|im_end|>','<|eot_id|>'],
#          control tokens only, which is exactly why the _qwen variant was made.
#   GATE1  `new slot, n_ctx` / `n_slots` readback == intended per-slot / parallel (bug-597:
#          llama_ctx is the TOTAL pool, divided by --parallel).
#   GATE2  summary.json.sampler.name == recommended, so this cell can never be silently
#          pooled with a greedy cohort.
set -u
export CUDA_VISIBLE_DEVICES=1          # bs2 GPU1 is ours. GPU0 is NOT.
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
export HF_ALLOW_CODE_EVAL=1            # bug-594: GATE0 imports tasks in THIS shell
command -v lm-eval >/dev/null || { echo "REFUSING: lm-eval not on PATH"; exit 1; }

OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
WORK=/mnt/sdc/ream-work
RES=/srv/ml/eval_results/qwen_suite
PORT=8099
SAMPLER=recommended
PROFILE=qwen3_6
B=arc_challenge_100
SLOT=24576
PAR=2
TOTAL=$(( SLOT * PAR ))
WAIT_MAX_S=${WAIT_MAX_S:-79200}        # 22 h: gate9d's remaining benches + all of gate10b
POLL_S=180

say(){ echo "[gate9e $(date -u +%H:%M:%S)Z] $*"; }

# arm | served-name | gguf | tokenizer   (identical to gate9d)
ARMS=(
  "pub|qwencodermpe_q6k|/srv/ml/models/gguf/Qwen3.6-35B-A3B-184e-coder-lcbmpe-GGUF/Qwen3.6-35B-A3B-184e-coder-lcbmpe-Q6_K.gguf|/srv/ml/models/Qwen3.6-35B-A3B"
  "armJ|qwenhybridp24_q6k|$WORK/gguf/armJ_imat/armJ-Q6_K.gguf|$WORK/armJ"
)

# ---- preflight: template parses, weights present ----
[ -s "$OMK/eval/templates/$B.yaml" ] || { say "REFUSE: template $B.yaml not installed"; exit 1; }
"$OMKPY" - "$OMK/eval/templates" "$B" <<'PY' || { say "GATE9E_REFUSE: template still does not load"; exit 3; }
import sys
sys.path.insert(0, sys.argv[1])
import template_loader as T
t = T.load(sys.argv[2])
print("  template ok: task=%s n=%s selection=%s" % (t.get("task"), t.get("n"), t.get("selection")))
PY
for a in "${ARMS[@]}"; do
  IFS='|' read -r arm name g tok <<<"$a"
  [ -s "$g" ]   || { say "REFUSE: $arm gguf missing: $g"; exit 1; }
  [ -d "$tok" ] || { say "REFUSE: $arm tokenizer missing: $tok"; exit 1; }
  say "preflight ok: $arm -> $(basename "$g")"
done

# ---- GATE 0 (FATAL): stop-word taxonomy ----
say "GATE0: resolving until= for $B"
"$OMKPY" - "$OMK/eval" "$B" <<'PY' || { say "GATE9E_REFUSE: GATE0 stop-word check failed"; exit 3; }
import sys, yaml, os
from lm_eval.tasks import TaskManager
base, b = sys.argv[1], sys.argv[2]
BAD = ("Question:", "Problem:", "Answer:", "\n\n")
t = yaml.safe_load(open(os.path.join(base, "templates", b + ".yaml")))
task = t["task"]
inc = (t.get("backend_args") or {}).get("lm_eval_include_path")
tm = TaskManager(include_path=os.path.join(base, inc)) if inc else TaskManager()
if task not in tm.all_tasks:
    print("  %s task=%s NOT REGISTERED" % (b, task)); print("GATE0_FAIL"); sys.exit(1)
gk = tm.load_task_or_group([task])[task].config.generation_kwargs or {}
u = gk.get("until") or []
hit = [s for s in u if s in BAD]
print("  %-20s task=%-30s until=%s %s" % (b, task, u, ("<<< POISONED " + str(hit)) if hit else "ok"))
print("GATE0_OK" if not hit else "GATE0_FAIL")
sys.exit(1 if hit else 0)
PY
say "GATE0 passed: no plain-text delimiter"

# ---- readiness: gate10b finished AND GPU1 actually free ----
# `pgrep -c` PRINTS 0 and EXITS 1 on no-match; `|| echo 0` would append a SECOND zero and
# the compare could never be true (bug-601). Use `|| true`.
t0=$(date +%s)
while :; do
  d=0; grep -aq "GATE10B_DONE" "$WORK/gate10b_loop.log" 2>/dev/null && d=1
  srv=$(pgrep -c -f "llama-server" 2>/dev/null || true); srv=${srv:-0}
  lme=$(pgrep -c -f "bin/lm-eval" 2>/dev/null || true); lme=${lme:-0}
  free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || true)
  if [ "$d" = 1 ] && [ "$srv" = 0 ] && [ "$lme" = 0 ] && [ "${free:-0}" -gt 80000 ]; then
    say "READY: gate10b done, no server/lm-eval alive, GPU1 free=${free}MiB"; break
  fi
  el=$(( $(date +%s) - t0 ))
  if [ "$el" -ge "$WAIT_MAX_S" ]; then
    say "REFUSE: not ready after ${el}s (gate10b_done=$d server=$srv lm_eval=$lme gpu1_free=${free}MiB)"
    exit 2
  fi
  [ $(( el % 3600 )) -lt "$POLL_S" ] && say "waiting ${el}s (gate10b_done=$d server=$srv lm_eval=$lme gpu1_free=${free}MiB)"
  sleep "$POLL_S"
done

ran=0; failed=0; aborted=0
for a in "${ARMS[@]}"; do
  IFS='|' read -r arm NAME G TOK <<<"$a"
  cell="$RES/$B/$NAME"
  [ -f "$cell/summary.json" ] && { say "SKIP $B/$arm (scored cell exists)"; continue; }
  # A dir with no summary.json is an interrupted/aborted run. PRESERVE it, never delete --
  # its sqlite cache makes the re-run resumable and its server.log is geometry evidence.
  if [ -d "$cell" ] && [ ! -d "$cell/sqlite_cache" ]; then
    mv "$cell" "${cell}_ABORTED_$(date -u +%Y%m%dT%H%M%SZ)" && say "preserved aborted $B/$arm"
  elif [ -d "$cell" ]; then
    say "REUSING $B/$arm partial cell (sqlite cache present -> resumable)"
  fi

  say "===== $B / $arm  per_slot=$SLOT par=$PAR total=$TOTAL sampler=$SAMPLER"
  "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$B" --quant q6_k \
      --model "$G" --tokenizer "$TOK" --served-name "$NAME" --port "$PORT" \
      --results-dir "$RES" --parallel "$PAR" \
      --sampler-profile "$PROFILE" --sampler "$SAMPLER" \
      --metadata backend_args.llama_ctx=$TOTAL
  rc=$?
  say "<<<< END $B/$arm rc=$rc"

  # ---- GATE 1: geometry readback ----
  L="$cell/server.log"
  got=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  sl=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  say "GEOMETRY $B/$arm per_slot=${got:-unknown} slots=${sl:-unknown} (want $SLOT / $PAR)"
  if [ "${got:-0}" != "$SLOT" ] || [ "${sl:-0}" != "$PAR" ]; then
    say "GATE9E_ABORT $B/$arm: geometry not honoured"; aborted=$((aborted+1)); continue; fi

  s="$cell/summary.json"
  [ -f "$s" ] || { say "FAIL $B/$arm: no summary.json"; failed=$((failed+1)); continue; }

  # ---- GATE 2: sampler provenance ----
  sn=$("$OMKPY" - "$s" <<'PY'
import json,sys
print(((json.load(open(sys.argv[1])).get("sampler") or {}).get("name")) or "NONE")
PY
)
  say "SAMPLER $B/$arm recorded=$sn (want $SAMPLER)"
  [ "$sn" = "$SAMPLER" ] || { say "GATE9E_ABORT $B/$arm: sampler mismatch"; aborted=$((aborted+1)); continue; }

  sc=$("$OMKPY" - "$s" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print("score=%s metric=%s filter=%s" % (d.get("score"), d.get("metric"), d.get("filter")))
PY
)
  say "SCORE $B/$arm $sc"
  ran=$((ran+1))
done

say "=== GATE9E_DONE ran=$ran failed=$failed aborted=$aborted ==="
say "REPORT THIS CELL AS n=100, never beside a full-1172 ARC figure (different batch"
say "composition = different basis). It IS a valid paired armJ-vs-pub comparison."
