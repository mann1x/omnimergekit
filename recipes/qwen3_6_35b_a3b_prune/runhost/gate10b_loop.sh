#!/usr/bin/env bash
# CoderX gate 2 RE-RUN (gate10b) — 48-seed agentic loop gate, PAIRED: anchor (pub) then armJ.
#
# WHY THIS EXISTS. gate10 (2026-08-20 15:25Z) produced 92-98% "runaway" on ALL FOUR cells
# (pub/armJ x qwen_deploy/qwen_vendor_bare). That was SAT_COLLAPSE, not model behaviour:
#   server ran -c 192000 --parallel 8  ->  new slot, n_ctx = 24064
#   fixture prompt                     ->  ~23,997 tok  (messages 5,556 + tools 89 defs 18,441)
#   generation headroom                ->  ~67 tok, against max_tokens 16384
# Every seed hit finish=length instantly and the harness books that as RUNAWAY. The one pass
# finished tool_calls because its output happened to fit. Invalid results are PRESERVED at
# results/QWEN_GATE10_SATCOLLAPSE_INVALID_<ts>/ with a REASON.md. Nothing was deleted.
#
# The old profile comment sized the window from the MESSAGES only ("~5.5k prompt"). That
# figure was right; the 18.4k tool-schema block -- 3.3x the messages -- was simply omitted.
# THE LESSON: for a tool-calling fixture the tool schemas ARE most of the prompt. Never size
# an agentic window from the message text.
#
# TWO FATAL PREFLIGHT GATES (this is the actual fix -- gate the taxonomy, not the last bug):
#   GATE-P  PROMPT BUDGET -- really tokenize messages+tools through EACH arm's own tokenizer
#           via apply_chat_template(..., tools=...). Refuse unless
#           per_slot >= prompt + max_tokens + MARGIN. No estimate, no chars/4.
#   GATE-G  GEOMETRY READBACK -- after the run, `new slot, n_ctx = N` and `n_slots = M` from
#           the arm's own server log must equal the intended per-slot / parallel
#           (bug-597: llama.cpp divides -c by --parallel; llama_ctx is the TOTAL pool).
#
# Each arm now writes to its OWN out_dir. In gate10 both arms wrote
# result_embedded__<fixture>.json + summary.json to one fixed path, so armJ silently
# overwrote pub's artifacts; only the log preserved both numbers.
set -u
export CUDA_VISIBLE_DEVICES=1          # bs2 GPU1 is ours; GPU0 is NOT. Export, never poll.
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
H=/srv/ml/repos/omnimergekit/tools/agentic-loop-harness
PY=/root/anaconda3/envs/omnimergekit/bin/python
WORK=/mnt/sdc/ream-work
FIX=$H/fixtures/solar_build_start.json
PER_SLOT=45056          # ctx_size 360448 / parallel 8
PARALLEL=8
MAXTOK=16384
MARGIN=2048
WAIT_MAX_S=${WAIT_MAX_S:-43200}
POLL_S=120

say(){ echo "[gate10b $(date -u +%H:%M:%S)Z] $*"; }

[ -d "$H/agentic_loop_harness" ] || { say "REFUSE: harness not at $H"; exit 1; }
[ -s "$FIX" ] || { say "REFUSE: fixture missing $FIX"; exit 1; }

# arm | profile | tokenizer dir | out_dir
ARMS=(
  "pub|loopgate_qwen_pub.yaml|/srv/ml/models/Qwen3.6-35B-A3B|/srv/ml/agentic_loop/results/gate10b_pub"
  "armJ|loopgate_qwen_armJ.yaml|$WORK/armJ|/srv/ml/agentic_loop/results/gate10b_armJ"
)

# ---------------- GATE-P: real tokenized prompt budget, per arm ----------------
say "GATE-P: tokenizing fixture (messages + tools) through each arm's own tokenizer"
badp=0
for spec in "${ARMS[@]}"; do
  IFS='|' read -r arm prof tok out <<< "$spec"
  [ -s "$WORK/$prof" ] || { say "GATE-P REFUSE $arm: missing profile $WORK/$prof"; badp=$((badp+1)); continue; }
  n=$("$PY" - "$FIX" "$tok" <<'PY' 2>/dev/null
import json, sys
from transformers import AutoTokenizer
fix, tokdir = sys.argv[1], sys.argv[2]
d = json.load(open(fix))
tk = AutoTokenizer.from_pretrained(tokdir, trust_remote_code=True)
out = tk.apply_chat_template(d["messages"], tools=d.get("tools"),
                             add_generation_prompt=True, tokenize=True)
# transformers 5.x returns a BatchEncoding, so a bare len() counts its 2 KEYS
# (input_ids, attention_mask) and reports "2". That is how the first cut of this
# gate PASSED on prompt=2 tok. Always unwrap to the actual id list.
if hasattr(out, "keys"):
    out = out["input_ids"]
while isinstance(out, (list, tuple)) and out and isinstance(out[0], (list, tuple)):
    out = out[0]
print(len(out))
PY
)
  if ! [ "${n:-}" -gt 0 ] 2>/dev/null; then
    say "GATE-P REFUSE $arm: could not tokenize fixture with $tok"; badp=$((badp+1)); continue
  fi
  # SANITY FLOOR: a 4-message + 89-tool agentic fixture cannot tokenize under 1000.
  # A check that green-lights on an absurd number is a BROKEN check, not a passing one.
  if [ "$n" -lt 1000 ]; then
    say "  GATE-P REFUSE $arm: prompt=${n} tok is impossible for this fixture (89 tool defs)."
    say "  The TOKENIZATION is broken, not the budget. Refusing rather than passing on garbage."
    badp=$((badp+1)); continue
  fi
  need=$(( n + MAXTOK + MARGIN ))
  say "  $arm prompt=${n} tok + max_tokens=${MAXTOK} + margin=${MARGIN} => need ${need}/slot, have ${PER_SLOT}"
  if [ "$PER_SLOT" -lt "$need" ]; then
    say "  GATE-P REFUSE $arm: per-slot ${PER_SLOT} < ${need}. THIS IS THE gate10 SAT_COLLAPSE. Not launching."
    badp=$((badp+1))
  fi
done
[ "$badp" -eq 0 ] || { say "=== GATE10B_ABORT: GATE-P failed for $badp arm(s) ==="; exit 3; }
say "GATE-P passed: every arm has real generation headroom"

# ---------------- readiness: gate9d finished AND GPU1 actually free ----------------
t0=$(date +%s)
while :; do
  d9d=0; grep -aq "GATE9D_DONE" "$WORK/gate9d_gapfill.log" 2>/dev/null && d9d=1
  # NOTE: `pgrep -c` PRINTS 0 and EXITS 1 on no-match; `|| echo 0` would append a SECOND
  # zero and the compare could never be true (bug-601). Use `|| true`.
  srv=$(pgrep -c -f "llama-server" 2>/dev/null || true); srv=${srv:-0}
  lme=$(pgrep -c -f "bin/lm-eval" 2>/dev/null || true); lme=${lme:-0}
  free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || true)
  if [ "$d9d" = 1 ] && [ "$srv" = 0 ] && [ "$lme" = 0 ] && [ "${free:-0}" -gt 80000 ]; then
    say "READY: gate9d done, no server/lm-eval alive, GPU1 free=${free}MiB"; break
  fi
  el=$(( $(date +%s) - t0 ))
  if [ "$el" -ge "$WAIT_MAX_S" ]; then
    say "REFUSE: not ready after ${el}s (gate9d_done=$d9d server=$srv lm_eval=$lme gpu1_free=${free}MiB)."
    exit 2
  fi
  [ $(( el % 1800 )) -lt "$POLL_S" ] && say "waiting ${el}s (gate9d_done=$d9d server=$srv lm_eval=$lme gpu1_free=${free}MiB)"
  sleep "$POLL_S"
done

# ---------------- run: anchor first, then candidate ----------------
rc_all=0; aborted=0
for spec in "${ARMS[@]}"; do
  IFS='|' read -r arm prof tok out <<< "$spec"
  mkdir -p "$out"
  say "===== LOOPGATE $arm profile=$prof out=$out"
  PYTHONPATH=$H "$PY" -m agentic_loop_harness --profile "$WORK/$prof" 2>&1 | sed "s/^/[$arm] /"
  rc=${PIPESTATUS[0]}
  say "<<<< END $arm rc=$rc"
  [ "$rc" = 0 ] || rc_all=1

  # ---- GATE-G: geometry readback (the gate10 failure, now fatal) ----
  L=$(ls -1t "$out"/*server*.log 2>/dev/null | head -1)
  if [ -z "$L" ]; then
    say "GATE10B_ABORT $arm: no server log under $out -- geometry unverifiable"; aborted=$((aborted+1)); continue
  fi
  got=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" | head -1 | grep -oE "[0-9]+$")
  sl=$(grep -aoE "n_slots = [0-9]+" "$L" | head -1 | grep -oE "[0-9]+$")
  say "GEOMETRY $arm per_slot=${got:-unknown} slots=${sl:-unknown} (want $PER_SLOT / $PARALLEL)"
  if [ "${got:-0}" != "$PER_SLOT" ] || [ "${sl:-0}" != "$PARALLEL" ]; then
    say "GATE10B_ABORT $arm: geometry not honoured -- this cell is SAT_COLLAPSE-shaped, NOT a result"
    aborted=$((aborted+1))
  fi
done

say "=== GATE10B_DONE rc_all=$rc_all aborted=$aborted ==="
say "PASS CRITERION is PAIRED, not absolute: armJ's loop/runaway rate must not exceed the"
say "published cut's by more than the harness's own seed noise. A shared high rate is a"
say "fixture property; a gap is an armJ property. Do not read armJ's 48 seeds alone."
say "If BOTH arms are again >90%, do NOT report it as a model finding -- re-check GATE-P/GATE-G."
