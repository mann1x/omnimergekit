#!/usr/bin/env bash
# R9 (#845/#847, GEPO leg) -- shorten CoderX on LiveCodeBench without breaking it.
#
# BOTH bs2 GPUs, DDP, rollouts generated IN-PROCESS by HF transformers.
#
# WHY THERE IS NO vLLM SERVER HERE
# --------------------------------
# The predecessor (r9_gepo.sh) split the job GPU0=vLLM rollout server /
# GPU1=trainer. vLLM 0.20.2 ships Qwen3.5's text classes but wires none of the
# machinery around them, so serving armJ required a hand-written architecture
# wrapper patching four upstream gaps. Three were provably correct (the weight
# gate reports expected=573 loaded=573 missing=0); the fourth -- a hand-rolled
# M-RoPE position path -- silently emitted token soup. Result on 2026-08-23:
# every one of 32 rollouts scored 0.0, grad_norm was 0 on every step, and the
# smoke still printed GEPO_TRAIN_DONE. The 18h full leg would have produced an
# untrained adapter.
#
# Generating through the same transformers object that is being trained removes
# that entire failure class: there is no second engine that can disagree with the
# one under training, and the trainer-vs-sampler logp gap (measured 7.7 nats,
# importance ratio 4e-23) is zero by construction.
#
# THE GATE IS grad_norm, NOT THE SENTINEL
# ---------------------------------------
# "Did the process finish" is not "did it learn". This chain refuses to launch
# the full leg unless the smoke shows a NON-ZERO grad_norm and a NON-ZERO
# reward_std -- the two numbers that were flat zero while the old gate passed.
set -uo pipefail

REPO=/srv/ml/repos/omnimergekit
PY=/mnt/sdc/ml/brevity/gepo/venv/bin/python
W=/mnt/sdc/ml/brevity/gepo
MODEL=/mnt/sdc/ream-work/armJ

BUDGET=${BUDGET:-12288}
LAMBDA=${LAMBDA:-0.7}
MAXCOMP=${MAXCOMP:-16384}
G=${G:-8}
ACCUM=${ACCUM:-16}          # must be a multiple of G -- groups cannot span ranks
LR=${LR:-5e-6}
BETA=${BETA:-0.02}
EPOCHS=${EPOCHS:-1}
SKIP_SMOKE=${SKIP_SMOKE:-0}
# 0 = whole pool. load_pool() takes rows[:limit], i.e. the FIRST n rows -- which is
# only an unbiased sample because lcb_rl_pool.jsonl is already shuffled (verified
# 2026-08-23: ids run 2808, 3246, 3555, 2757, 3511 ... with no ordering). If that
# file is ever regenerated in sorted order, cutting it here becomes a biased draw.
POOL_LIMIT=${POOL_LIMIT:-0}
# Checkpoint often. A short run has few steps, and the 2026-08-23 41h projection
# showed how much it costs to have nothing on disk when a run has to be stopped.
SAVE_STEPS=${SAVE_STEPS:-8}
# Adapters are NEVER deleted. This was hardcoded to run1, so a second invocation
# would have trained straight over the shipped a3b-gepo1 adapter. Parametrised, and
# the run refuses if the directory already exists.
RUN=${RUN:-run1}
# run3 (#854): ALSO LoRA the MoE router. Default 0 so run1/run2 reproduce from
# this file unchanged. Turning it on forces dropout 0 inside the trainer
# (lora.ParamWrapper constraint) -- a basis change that gepo_brevity.py logs and
# records in run_meta.json. See the LORA_REGEX comment block there.
ROUTER_LORA=${ROUTER_LORA:-0}
# run4: reward version + capability replay. Defaults keep run1-run3 reproducible from
# this file: REWARD=v1 and an empty REPLAY_POOL reproduce the old command line exactly.
REWARD=${REWARD:-v1}
REPLAY_POOL=${REPLAY_POOL:-}
REPLAY_N=${REPLAY_N:-0}
REPLAY_BALANCE=${REPLAY_BALANCE:-equal}
REPLAY_NO_THINK=${REPLAY_NO_THINK:-0}
REPLAY_DIFFICULTY=${REPLAY_DIFFICULTY:-}
# run4 (rebalanced): ONE mixed pool carrying all four tiers, replacing the two-file
# --pool + --replay-pool shape. Empty keeps gepo_brevity's own default (the LCB pool),
# so run1-run3 still reproduce from this file unchanged.
POOL=${POOL:-}
# Sampling temperature. AN/Gemma-4-E2B's working GEPO ran 0.9; r9 run1-3 ran 0.6.
TEMPERATURE=${TEMPERATURE:-}

ts(){ date '+%F %T %Z'; }
say(){ echo "[$(ts)] $*"; }

SCOPE_ARGS=()
if [ "$ROUTER_LORA" = "1" ]; then SCOPE_ARGS+=(--router-lora); fi
SCOPE_ARGS+=(--reward "$REWARD")
if [ -n "$POOL" ]; then
  [ -s "$POOL" ] || { echo "FATAL: POOL=$POOL missing/empty"; exit 7; }
  SCOPE_ARGS+=(--pool "$POOL")
fi
if [ -n "$TEMPERATURE" ]; then SCOPE_ARGS+=(--temperature "$TEMPERATURE"); fi
# The contamination gate belongs on whichever file carries the GPQA/MBPP rows. In the
# two-file shape that is REPLAY_POOL; in the mixed shape the replay tiers live INSIDE
# POOL and gating REPLAY_POOL alone would gate nothing at all.
GATE_POOL="${REPLAY_POOL:-}"
if [ -z "$GATE_POOL" ] && [ -n "$POOL" ]; then GATE_POOL="$POOL"; fi
if [ -n "$GATE_POOL" ]; then
  # Contamination gate on the ARTIFACT, on THIS host, before any GPU time is spent.
  # The build-time gate (test_gepo_replay_gate.py) runs only where the pool is BUILT
  # -- it needs gpqa_main.csv, which the run host does not have. So the file that
  # actually gets trained on has, until this point, been certified only by a sha256
  # match against a note. This re-derives the three zeros from the eval sets present
  # here, with a fire-control on each gate. A leak makes every downstream cell of
  # GPQA / MBPP / HumanEval a lie, so it aborts rather than warns.
  echo "== replay artifact gate =="
  "$PY" "$(dirname "$0")/gate_replay_artifact.py" --pool "$GATE_POOL" \
    || { echo "FATAL: $GATE_POOL failed the artifact contamination gate"; exit 9; }
fi
if [ -n "$REPLAY_POOL" ]; then
  SCOPE_ARGS+=(--replay-pool "$REPLAY_POOL" --replay-n "$REPLAY_N")
  SCOPE_ARGS+=(--replay-balance "$REPLAY_BALANCE")
  if [ "$REPLAY_NO_THINK" = "1" ]; then SCOPE_ARGS+=(--replay-no-think); fi
fi
# OUTSIDE the REPLAY_POOL branch: the mixed pool has no separate replay file, and
# leaving this inside would have accepted REPLAY_DIFFICULTY, recorded it in
# run_meta.json, and filtered nothing. [[feedback_a_declared_gate_is_not_a_wired_gate]]
if [ -n "$REPLAY_DIFFICULTY" ]; then
  [ -s "$REPLAY_DIFFICULTY" ] || { echo "FATAL: REPLAY_DIFFICULTY=$REPLAY_DIFFICULTY missing/empty"; exit 10; }
  SCOPE_ARGS+=(--replay-difficulty "$REPLAY_DIFFICULTY")
fi

launch(){   # launch <name> <out> <extra args...>
  local name="$1" out="$2"; shift 2
  say "--- $name ---"
  CUDA_VISIBLE_DEVICES=0,1 "$PY" -m accelerate.commands.launch \
      --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no \
      "$REPO/scripts/gepo_brevity.py" \
      --model "$MODEL" --out "$out" \
      --length-budget "$BUDGET" --length-lambda "$LAMBDA" \
      --beta "$BETA" --lr "$LR" \
      ${SCOPE_ARGS[@]+"${SCOPE_ARGS[@]}"} "$@" 2>&1 | tee "$W/${name}.log"
  return "${PIPESTATUS[0]}"
}

# Reads the LAST logged step and refuses on all-zero learning signal. Parsing the
# printed dict is deliberate: those are the numbers a human would read back.
gate_learned(){
  local log="$1"
  "$PY" - "$log" <<'PY'
import ast, re, sys
rows = []
for line in open(sys.argv[1], errors="ignore"):
    line = line.strip()
    if line.startswith("{") and "'grad_norm'" in line:
        try:
            rows.append(ast.literal_eval(line))
        except Exception:
            pass
if not rows:
    sys.exit("GATE_FAIL: no logged training steps at all")
gn = [float(r.get("grad_norm", 0) or 0) for r in rows]
rs = [float(r.get("reward_std", 0) or 0) for r in rows]
print(f"GATE grad_norm max={max(gn):.4g} | reward_std max={max(rs):.4g} "
      f"over {len(rows)} step(s)")
if max(gn) == 0:
    sys.exit("GATE_FAIL: grad_norm was 0 on EVERY step -- nothing was learned. "
             "This is the 2026-08-23 garbage-rollout signature; do not run the "
             "full leg.")
if max(rs) == 0:
    sys.exit("GATE_FAIL: reward_std was 0 on every step -- no within-group spread, "
             "so GEPO has no gradient to build even if rewards are non-zero.")
print("GATE_OK")
PY
}

# The training log IS the record of the run -- run1's step_time / mean_length /
# clipped_ratio series was read back out of it to size run2. It must not be
# overwritten either. run1's original logs are $W/full.log and $W/smoke_ddp.log
# (written before this was parametrised); everything after is suffixed by RUN.
if [ "$RUN" = "run1" ]; then FULL_LOG=full; SMOKE_LOG=smoke_ddp
else FULL_LOG="full_$RUN"; SMOKE_LOG="smoke_ddp_$RUN"; fi

# ROUTER gate (run3+). gate_learned reads the TOTAL grad_norm, which is non-zero
# from the 310 Linear targets alone -- so it CANNOT tell "router trained" from
# "router wrapped but receiving no gradient". LoRA B initialises to exactly zero,
# so a still-all-zero router lora_B after the smoke proves no gradient reached it.
# That is the whole point of run3, so it is fatal, not a warning.
gate_router_moved(){
  local adapter="$1"
  "$PY" - "$adapter" <<'PY'
import sys, pathlib
from safetensors import safe_open
p = pathlib.Path(sys.argv[1]) / "adapter_model.safetensors"
if not p.is_file():
    sys.exit(f"ROUTER_GATE_FAIL: no adapter at {p}")
n_b = moved = 0
with safe_open(str(p), framework="pt") as f:
    for k in f.keys():
        if "mlp.gate" in k and "lora_B" in k:
            n_b += 1
            if f.get_tensor(k).abs().sum().item() > 0:
                moved += 1
if n_b == 0:
    sys.exit("ROUTER_GATE_FAIL: adapter contains NO router lora_B tensors -- "
             "target_parameters did not wrap mlp.gate.weight at all.")
print(f"ROUTER_GATE router lora_B tensors={n_b} moved-off-zero={moved}")
if moved == 0:
    sys.exit("ROUTER_GATE_FAIL: every router lora_B is still exactly zero -- the "
             "router was wrapped but received NO gradient. Do not run the full "
             "leg; the router arm would be a no-op dressed as an experiment.")
print("ROUTER_GATE_OK")
PY
}

# REPLAY TIER gate (run4+). gate_learned reads the TOTAL grad_norm, which the LCB tier
# alone keeps non-zero -- so it CANNOT see a replay tier that scored 0.0 for every
# rollout because its reward_kind was unwired, its gold column was missing, or its
# verifier never matched. Such a tier has no within-group spread, contributes no
# gradient, and drags the policy for the whole run while every aggregate looks normal.
# The v2 reward prints per-tier n/mean/nz; this asserts every tier produced at least
# one non-zero reward.
gate_replay_tiers(){
  local log="$1"
  "$PY" - "$log" <<'PYGATE'
# Aggregate the UNION of every V2_REWARD tier line, across every rank.
#
# bug-641 (2026-08-28): this gate used to read only the LAST such line, and aborted
# run4's smoke claiming "only 1 tier scored" when two had scored and a third had
# scored but was never logged. The last line is not the population, for two
# compounding reasons:
#   * the reward's per-tier state is PER-RANK and DDP runs world=2, so any one line
#     is one rank's partial view -- rank0 had seen mbpp_exec, rank1 lcb_exec;
#   * the emitter SAMPLES on log_every, so a tier can score and never print.
# The emitter now also fires on the first sighting of a new reward_kind, so every
# tier that ever scored appears at least once; this side takes the union so a tier
# seen on either rank counts. [[feedback_one_log_line_is_not_a_cross_arm_reading]]
import re, sys
agg, lines = {}, 0
for ln in open(sys.argv[1], errors="ignore"):
    if ">>> V2_REWARD" not in ln or "| TIERS " not in ln:
        continue
    # [\w/]+ not \w+ : tier keys carry a /T|/N thinking suffix, and \w stops at the
    # slash -- it would capture the bare letter "T" as the whole tier name.
    found = re.findall(r"([\w/]+):n=(\d+),mean=([-\d.]+),nz=(\d+)",
                       ln.split("| TIERS ", 1)[1])
    if not found:
        continue
    lines += 1
    for k, n, m, z in found:
        # Per-rank counters are cumulative, so keep the MAX per rank-line rather than
        # summing across lines of the same rank (which would double-count).
        d = agg.setdefault(k, {"n": 0, "nz": 0})
        d["n"] = max(d["n"], int(n))
        d["nz"] = max(d["nz"], int(z))
if not agg:
    sys.exit("REPLAY_GATE_FAIL: no V2_REWARD tier line in the smoke log -- either the "
             "v2 reward never ran or it never reached its log interval.")

# EXPECTED tiers come from the mixer's own census, not from a hardcoded count. The
# dataset builder logs e.g.
#   REPLAY_MIX lcb=2 replay=4 {'mc_letter': 2, 'mbpp_exec': 2} -> 6 problems (...)
# so the gate can check OBSERVED against EXPECTED instead of against ">= 2". A bare
# count would have passed a log in which mc_letter was mixed in and never scored --
# asserting a tier alive that was never measured.
#
# POOL_CENSUS is the newer, universal form: the trainer logs it on EVERY path,
# including the single mixed pool (LCB + replay in one file) where no REPLAY_MIX
# line exists at all. Its keys carry a /T or /N thinking suffix that the reward's
# tier keys do not, so strip it. Both forms are read; the union is the expectation.
exp = set()
for ln in open(sys.argv[1], errors="ignore"):
    if "POOL_CENSUS" in ln:
        seg = ln.split("POOL_CENSUS", 1)[1]
        exp.update(re.findall(r'"(\w+(?:/[TN])?)":\s*[1-9]', seg))
    if "REPLAY_MIX" not in ln:
        continue
    seg = ln.split("REPLAY_MIX", 1)[1]
    if re.search(r"lcb=([1-9]\d*)", seg):
        exp.add("lcb_exec")
    exp.update(re.findall(r"'(\w+)':\s*[1-9]", seg))
# G is needed to turn a rollout count into a PROBLEM count. nz=0 over one problem is a
# hard problem; nz=0 over several is a dead tier, and the message must not conflate them.
G = 0
for ln in open(sys.argv[1], errors="ignore"):
    m = re.search(r"GROUPS_OK.*\(G=(\d+)\)", ln)
    if m:
        G = int(m.group(1))
print("REPLAY_TIERS(union over %d line(s), G=%s) " % (lines, G or "?")
      + " ".join("%s(n=%d,nz=%d,problems=%s)"
                 % (k, d["n"], d["nz"], (d["n"] // G) if G else "?")
                 for k, d in sorted(agg.items()))
      + "  EXPECTED=%s" % (sorted(exp) or "<no census line>"))
# The reward tags tiers as reward_kind/T|N; REPLAY_MIX (older, replay-pool path) knows
# only the bare kind. Match a SUFFIXED expectation exactly and a BARE one against any
# thinking mode, so the two census forms can be compared against one observed set
# without either producing a phantom miss.
base = lambda k: k.split("/")[0]
obs_full, obs_base = set(agg), {base(k) for k in agg}
dead = [k for k, d in agg.items() if d["nz"] == 0]
missing = sorted(e for e in exp
                 if not ((e in obs_full) if "/" in e else (base(e) in obs_base)))
if missing:
    sys.exit("REPLAY_GATE_FAIL: tier(s) %s were MIXED INTO the pool but never scored a "
             "single rollout. A tier that never reaches the reward is indistinguishable "
             "from a dead one, and it would burn the run producing no gradient." % missing)
if not exp and len(agg) < 2:
    sys.exit("REPLAY_GATE_FAIL: only %d tier(s) scored and no REPLAY_MIX census was "
             "logged, so the expected set is unknown. Refusing on an unprovable "
             "basis." % len(agg))
if dead:
    # SINGLE-PROBLEM nz=0 is not evidence of a dead tier. A GRPO group is G rollouts of
    # ONE prompt, so nz=0 over one problem says only "the model got that problem wrong",
    # which for a hard GPQA item is the expected outcome, not a wiring fault. Aborting a
    # 41h run on it is a false positive. Over several problems it is a real signal.
    # [[feedback_gate_scope_must_match_sample_size]]
    thin = [k for k in dead if G and agg[k]["n"] // G < 2]
    if thin:
        sys.exit("REPLAY_GATE_FAIL: tier(s) %s scored zero, but each was backed by only "
                 "ONE problem at G=%d -- that cannot distinguish a dead tier from a hard "
                 "problem, so the gate refuses to rule either way. Re-run the smoke with "
                 "--limit-per-tier >= 2." % (thin, G))
    sys.exit("REPLAY_GATE_FAIL: tier(s) %s produced ZERO non-zero rewards across %s "
             "problems each -- no within-group spread, no gradient, and they would drag "
             "the policy for the whole run while grad_norm stays healthy from the other "
             "tier." % (dead, {k: agg[k]["n"] // G for k in dead} if G else "?"))
print("REPLAY_TIER_ALIVE_OK")
PYGATE
}

if [ "$SKIP_SMOKE" = "0" ]; then
  # --replay-n 4 (argparse takes the LAST occurrence, overriding SCOPE_ARGS) keeps the
  # smoke small while still putting BOTH replay tiers through the reward -- the whole
  # point of the gate below. Without it the smoke would pull all $REPLAY_N replay rows
  # and stop being a smoke.
  SMOKE_EXTRA=()
  # The smoke's completion cap must be one the REPLAY tiers can actually finish under,
  # or the tier gate below measures the cap instead of the tier. 2026-08-28: run4's
  # second smoke ran at 2048 and both replay tiers came back clipped=1.000, pass=0.000,
  # nz=0 -- every GPQA chain-of-thought and every MBPP solution truncated before it
  # could emit an answer, so the verifier saw nothing and scored 0. lcb_exec survived
  # the same cap (nz=3) only because a truncated LCB passer is censored to 0.1 rather
  # than failed, which is exactly the asymmetry that makes a shared small cap
  # unreadable. [[feedback_cap_asymmetry_turns_a_bench_into_a_length_meter]]
  #
  # So when a replay pool is in play the smoke runs at the FULL budget. It costs real
  # minutes, and it buys the one thing the 25-30h run rests on: evidence that replay
  # rows CAN score at the budget the run will actually use. A cheaper smoke that
  # cannot answer that question is not cheaper, it is uninformative.
  SMOKE_MAXCOMP=2048
  SMOKE_LIMIT=(--limit 2)
  if [ -n "$REPLAY_POOL" ]; then SMOKE_EXTRA+=(--replay-n 4); SMOKE_MAXCOMP="$MAXCOMP"; fi
  # Mixed pool: draw one problem of EVERY tier instead of the first 2 rows. --limit 2
  # on a shuffled 4-tier pool most likely yields two rows of the same tier, and the
  # tier gate would then certify only the tiers that happened to be drawn -- passing
  # while saying nothing about the three it never saw.
  if [ -n "$POOL" ]; then
    # THREE problems per tier, not one. One problem cannot establish that a tier is
    # alive: a GRPO group is G rollouts of a SINGLE prompt, so a hard problem yields
    # nz=0 for reasons that have nothing to do with wiring. Measured 2026-08-28: the
    # first smoke of the mixed pool aborted on mc_letter nz=0 from exactly one hard
    # Biology item. Costs ~50 min instead of ~25 on a 41h run.
    SMOKE_LIMIT=(--limit-per-tier "${SMOKE_PER_TIER:-3}"); SMOKE_MAXCOMP="$MAXCOMP"
  fi
  launch "$SMOKE_LOG" "$W/$SMOKE_LOG" \
      --max-completion "$SMOKE_MAXCOMP" --num-generations 4 --grad-accum 4 \
      "${SMOKE_LIMIT[@]}" --epochs 1 --save-steps 1000 \
      ${SMOKE_EXTRA[@]+"${SMOKE_EXTRA[@]}"}
  gate_learned "$W/${SMOKE_LOG}.log" || { say "FATAL: smoke did not learn"; exit 4; }
  if [ "$ROUTER_LORA" = "1" ]; then
    gate_router_moved "$W/$SMOKE_LOG" || { say "FATAL: router got no gradient"; exit 6; }
  fi
  if [ -n "$REPLAY_POOL" ] || [ -n "$POOL" ]; then
    gate_replay_tiers "$W/${SMOKE_LOG}.log" || { say "FATAL: a replay tier is dead"; exit 8; }
  fi
  say "SMOKE_OK"
fi

[ -e "$W/$RUN" ] && { say "FATAL: $W/$RUN already exists -- refusing to train over an existing adapter. Set RUN=<newname>."; exit 2; }

launch "$FULL_LOG" "$W/$RUN" \
    --max-completion "$MAXCOMP" --num-generations "$G" --grad-accum "$ACCUM" \
    --limit "$POOL_LIMIT" --epochs "$EPOCHS" --save-steps "$SAVE_STEPS"
RC=$?
grep -q GEPO_TRAIN_DONE "$W/${FULL_LOG}.log" || { say "FATAL: full leg died (rc=$RC)"; exit 5; }
gate_learned "$W/${FULL_LOG}.log" || say "WARNING: full leg finished but the learning gate failed"

say "adapter at $W/$RUN"
echo "###### R9_GEPO_DONE $(ts) ######"
