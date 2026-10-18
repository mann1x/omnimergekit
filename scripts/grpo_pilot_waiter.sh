#!/usr/bin/env bash
# Waits for the LR pilot to finish, gates it, and launches the FULL arm only on a
# clean PASS. Runs ON bs2.
#
# FAILS CLOSED. The full arm is ~22.5 h on both GPUs; every ambiguous reading
# (gate says WAIT, trainer died early, clipped_ratio pinned at 1.0) stops here and
# writes a report instead of launching.
#
# Why the sentinel and not $?: grpo_kl_gate.py returns 0 for BOTH "GATE PASS" and
# "WAIT: only N logged steps". Keying on the exit code would read an unfinished run
# as a pass. [[feedback_gate_needs_a_sentinel_not_an_exit_code]]
set -uo pipefail

REPO=/srv/ml/repos/omnimergekit
PY=/srv/ml/envs/envs/omk-grpo/bin/python
PILOT=/mnt/sdc/v7rework/grpo_lr_pilot_lr1e-5
FULL=/mnt/sdc/v7rework/grpo_full_lr1e-5
REPORT=/mnt/sdc/v7rework/PILOT_VERDICT.txt
MIN_STEPS=30

log() { echo "$(date -u +%FT%TZ) $*"; }

log ">>> waiting for pilot to finish: $PILOT"
# Primary signal: the trainer processes exit. Secondary: enough logged kl steps.
while true; do
    alive=$(ps -eo args | grep -c "[t]rain_grpo_efficiency.*grpo_lr_pilot_lr1e-5")
    steps=$($PY - "$PILOT" <<'PYEOF' 2>/dev/null || echo 0
import json,os,sys
p=sys.argv[1]; best=0; f=None
for d in os.listdir(p) if os.path.isdir(p) else []:
    if d.startswith("checkpoint-"):
        c=os.path.join(p,d,"trainer_state.json")
        if os.path.isfile(c):
            try: n=int(d.split("-")[1])
            except ValueError: continue
            if n>=best: best,f=n,c
print(len([e for e in json.load(open(f))["log_history"] if "kl" in e]) if f else 0)
PYEOF
)
    log "    alive=$alive  kl_steps=$steps"
    [ "$alive" -eq 0 ] && { log "    trainer exited"; break; }
    [ "${steps:-0}" -ge "$MIN_STEPS" ] && { log "    reached $steps kl steps"; break; }
    sleep 120
done

sleep 30   # let the final checkpoint flush

log ">>> running KL gate"
GATE_OUT=$(cd "$REPO" && $PY scripts/grpo_kl_gate.py "$PILOT" --min-steps 25 2>&1)
echo "$GATE_OUT"

# --- decision, fail-closed -------------------------------------------------
VERDICT=BLOCK
REASON=""
if grep -q "GATE PASS" <<<"$GATE_OUT"; then
    VERDICT=PASS
elif grep -q "GATE FAIL" <<<"$GATE_OUT"; then
    REASON="KL flat or falling -- the grpo_an_short signature. Raise --lr or lower --beta."
elif grep -q "^WAIT" <<<"$GATE_OUT"; then
    REASON="gate returned WAIT: the pilot did not reach 25 logged steps. It died early."
else
    REASON="gate produced no recognised verdict line."
fi

# Independent check: a clipped_ratio pinned near 1.0 means TRL counted every
# completion truncated (the EOS scalar-compare caveat the runner itself printed),
# so 'length' would be measuring censorship, not brevity.
CLIP=$($PY - "$PILOT" <<'PYEOF' 2>/dev/null || echo NA
import json,os,sys,statistics as st
p=sys.argv[1]; best=0; f=None
for d in os.listdir(p) if os.path.isdir(p) else []:
    if d.startswith("checkpoint-"):
        c=os.path.join(p,d,"trainer_state.json")
        if os.path.isfile(c):
            try: n=int(d.split("-")[1])
            except ValueError: continue
            if n>=best: best,f=n,c
if not f: print("NA"); raise SystemExit
v=[e[k] for e in json.load(open(f))["log_history"]
   for k in e if "clip" in k.lower() and isinstance(e[k],(int,float))]
print(f"{st.median(v):.4f}" if v else "NA")
PYEOF
)
log "    clipped_ratio median = $CLIP"
if [ "$VERDICT" = PASS ] && [ "$CLIP" != NA ]; then
    if awk "BEGIN{exit !($CLIP > 0.95)}"; then
        VERDICT=BLOCK
        REASON="KL rose but clipped_ratio median=$CLIP (>0.95): TRL counted ~every completion truncated, so the length signal is measuring censorship. Fix the EOS accounting before spending 22.5 h."
    fi
fi

{
  echo "PILOT VERDICT $(date -u +%FT%TZ)"
  echo "pilot: $PILOT"
  echo "verdict: $VERDICT"
  [ -n "$REASON" ] && echo "reason: $REASON"
  echo "clipped_ratio median: $CLIP"
  echo "--- gate output ---"; echo "$GATE_OUT"
} > "$REPORT"
log ">>> verdict=$VERDICT  (report: $REPORT)"

if [ "$VERDICT" != PASS ]; then
    log ">>> NOT launching the full arm. $REASON"
    echo "PILOT_WAITER_DONE verdict=BLOCK"
    exit 0
fi

if [ -e "$FULL" ]; then
    log ">>> REFUSE: $FULL already exists -- never overwrite an arm"
    echo "PILOT_WAITER_DONE verdict=PASS_BUT_OUT_EXISTS"
    exit 0
fi

log ">>> GATE PASS -- launching FULL arm, 450 steps, same basis (cap 4096, beta 0.01, lambda 0.8, q 0.6)"
cd "$REPO"
LR=1e-5 STEPS=450 OUT="$FULL" WANDB_PROJECT=grpo-efficiency \
  nohup setsid bash scripts/run_grpo_lr_pilot.sh \
    > /mnt/sdc/v7rework/grpo_full_lr1e-5.launch.log 2>&1 < /dev/null &
sleep 90
grep -aE "WANDB (on|off)" /mnt/sdc/v7rework/grpo_full_lr1e-5.launch.log | head -2
echo "PILOT_WAITER_DONE verdict=PASS launched=$FULL"
