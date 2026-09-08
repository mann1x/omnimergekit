#!/usr/bin/env bash
# CoderX gate 2 — 48-seed agentic loop gate, PAIRED: published cut (anchor) then armJ.
#
# ORDER IS DELIBERATE: the anchor runs FIRST. If the published cut's loop rate can't be
# measured, armJ's number has no floor and the gate is worthless -- better to find that out
# before spending the second run. (feedback_anchor_on_the_untuned_base_not_the_previous_arm)
#
# READINESS, NOT SLEEP. GPU1 may still be held by gate9c. This script POLLS A PREDICATE --
# gate9c's own DONE sentinel + no live llama-server + actual free VRAM -- and refuses after
# the deadline rather than starting on a contended GPU. It never touches GPU0 (not ours).
#
# The harness is NOT pip-installed in the omnimergekit env; it runs from source via
# PYTHONPATH against the pinned bs2 clone. Do NOT `git pull` that clone.
set -u
export CUDA_VISIBLE_DEVICES=1
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
H=/srv/ml/repos/omnimergekit/tools/agentic-loop-harness
PY=/root/anaconda3/envs/omnimergekit/bin/python
WORK=/mnt/sdc/ream-work
OUT=/srv/ml/agentic_loop/results
WAIT_MAX_S=${WAIT_MAX_S:-43200}          # 12 h ceiling; gate9c is the long pole
POLL_S=120

say(){ echo "[gate10 $(date -u +%H:%M:%S)Z] $*"; }

# ---- preflight: everything that can be checked without a GPU ----
[ -d "$H/agentic_loop_harness" ] || { say "REFUSE: harness not at $H"; exit 1; }
[ -s "$WORK/qwen36_deploy.json" ] || { say "REFUSE: missing matrix $WORK/qwen36_deploy.json"; exit 1; }
for f in "$WORK/loopgate_qwen_pub.yaml" "$WORK/loopgate_qwen_armJ.yaml"; do
  [ -s "$f" ] || { say "REFUSE: missing profile $f"; exit 1; }
  g=$(grep -E "^  gguf:" "$f" | awk '{print $2}')
  [ -s "$g" ] || { say "REFUSE: $f points at a missing GGUF: $g"; exit 1; }
  say "preflight ok: $(basename "$f") -> $g ($(du -h "$g" | cut -f1))"
done
mkdir -p "$OUT"

# ---- readiness predicate: gate9c finished AND GPU1 actually free ----
t0=$(date +%s)
while :; do
  done9c=0; grep -aq "GATE9C_DONE" "$WORK/gate9c_armJ.log" 2>/dev/null && done9c=1
  srv=$(pgrep -c -f "llama-server" 2>/dev/null || true)
  lme=$(pgrep -c -f "bin/lm-eval" 2>/dev/null || true)
  free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || true)
  if [ "$done9c" = 1 ] && [ "$srv" = 0 ] && [ "$lme" = 0 ] && [ "${free:-0}" -gt 80000 ]; then
    say "READY: gate9c done, no server/lm-eval alive, GPU1 free=${free}MiB"; break
  fi
  el=$(( $(date +%s) - t0 ))
  if [ "$el" -ge "$WAIT_MAX_S" ]; then
    say "REFUSE: not ready after ${el}s (gate9c_done=$done9c llama_server=$srv lm_eval=$lme gpu1_free=${free}MiB). NOT starting on a contended GPU."
    exit 2
  fi
  [ $(( el % 1800 )) -lt "$POLL_S" ] && say "waiting ${el}s (gate9c_done=$done9c server=$srv lm_eval=$lme gpu1_free=${free}MiB)"
  sleep "$POLL_S"
done

# ---- run: anchor first, then candidate ----
rc_all=0
for spec in "pub:loopgate_qwen_pub.yaml" "armJ:loopgate_qwen_armJ.yaml"; do
  arm=${spec%%:*}; prof=$WORK/${spec#*:}
  say "===== LOOPGATE $arm  profile=$(basename "$prof")"
  PYTHONPATH=$H "$PY" -m agentic_loop_harness --profile "$prof" 2>&1 | sed "s/^/[$arm] /"
  rc=${PIPESTATUS[0]}
  say "<<<< END $arm rc=$rc"
  [ "$rc" = 0 ] || rc_all=1
done

say "=== GATE10_DONE rc_all=$rc_all — read the per-cell fail_rate out of $OUT ==="
say "PASS CRITERION is PAIRED, not absolute: armJ's loop/runaway rate must not exceed the"
say "published cut's by more than the harness's own seed noise. A shared high rate is a"
say "fixture property; a gap is an armJ property. Do not read armJ's 48 seeds alone."
