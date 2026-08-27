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

ts(){ date '+%F %T %Z'; }
say(){ echo "[$(ts)] $*"; }

SCOPE_ARGS=()
if [ "$ROUTER_LORA" = "1" ]; then SCOPE_ARGS+=(--router-lora); fi

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

if [ "$SKIP_SMOKE" = "0" ]; then
  launch "$SMOKE_LOG" "$W/$SMOKE_LOG" \
      --max-completion 2048 --num-generations 4 --grad-accum 4 \
      --limit 2 --epochs 1 --save-steps 1000
  gate_learned "$W/${SMOKE_LOG}.log" || { say "FATAL: smoke did not learn"; exit 4; }
  if [ "$ROUTER_LORA" = "1" ]; then
    gate_router_moved "$W/$SMOKE_LOG" || { say "FATAL: router got no gradient"; exit 6; }
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
