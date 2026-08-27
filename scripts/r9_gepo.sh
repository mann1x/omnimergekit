#!/usr/bin/env bash
# R9 (#845, GEPO leg) — shorten CoderX on LiveCodeBench without breaking it.
#
#   GPU0  vLLM rollout server  (scripts/vllm_serve_qwen35.py, `vllm` conda env)
#   GPU1  GEPOTrainer + LoRA   (scripts/gepo_brevity.py,      `omnimergekit` env)
#
# BOTH GPUs are used. GPU0 is normally not ours; it is in scope for this run only
# because it was explicitly authorised. Nothing here touches anything else on the box.
#
# The two envs are deliberate: `vllm` serves (vllm 0.20.2 / transformers 5.9.0) and
# `omnimergekit` trains (transformers 5.5.0). Their pins diverge and merging them
# breaks the canonical eval path.
#
# SMOKE FIRST, ALWAYS. The full run is ~1520 rollouts of up to 16k tokens each, every
# one of them EXECUTED against real LCB tests. A recipe error found at step 90 costs
# the whole run. The smoke leg does 4 problems end to end -- generate, execute, score,
# backward, save -- and the chain stops on it unless it prints GEPO_TRAIN_DONE.
set -uo pipefail

REPO=/srv/ml/repos/omnimergekit
# The trainer canNOT run on the `omnimergekit` env. TRL's server mode still imports
# vLLM in the TRAINER process -- `VLLMClient.init_communicator` needs
# vllm.distributed.device_communicators.pynccl (PyNcclCommunicator) +
# StatelessProcessGroup for the NCCL channel that pushes LoRA weights to the rollout
# server (trl/generation/vllm_client.py:41-42). That is a real dependency, not a
# cosmetic availability check. omnimergekit is torch 2.10.0+cu128 while vLLM 0.20.2 is
# built against 2.11.0+cu130, so installing vLLM there would fail on ABI -- and pulling
# vLLM's deps in would drag transformers 5.5.0 -> 5.9.0 and move the canonical eval
# basis. So the trainer runs on a `venv --system-site-packages` OVERLAY of the vllm
# env, which adds exactly one package (peft) in its own site-packages and cannot write
# into the vllm env at all. Verified 2026-08-23: `vllm` env still has no peft.
PY_TRAIN=/mnt/sdc/ml/brevity/gepo/venv/bin/python
PY_SERVE=/root/anaconda3/envs/vllm/bin/python
W=/mnt/sdc/ml/brevity/gepo
PORT=8477
SERVE_DIR=$W/armJ_serve
MODEL=/mnt/sdc/ream-work/armJ

BUDGET=${BUDGET:-12288}
LAMBDA=${LAMBDA:-0.7}
MAXCOMP=${MAXCOMP:-16384}
G=${G:-8}
ACCUM=${ACCUM:-16}
LR=${LR:-5e-6}
BETA=${BETA:-0.02}
EPOCHS=${EPOCHS:-1}
SKIP_SMOKE=${SKIP_SMOKE:-0}

mkdir -p "$W"
ts(){ date '+%F %T %Z'; }
say(){ echo "[$(ts)] $*"; }

# ------------------------------------------------------------------ preflight
say "=== R9 preflight ==="
for f in "$MODEL/config.json" "$REPO/eval/lcb/lcb_rl_pool.jsonl" \
         "$REPO/scripts/gepo_brevity.py" "$REPO/scripts/gepo_trainer.py" \
         "$REPO/scripts/lcb_brevity_reward.py" "$REPO/scripts/vllm_serve_qwen35.py"; do
  [ -e "$f" ] || { say "FATAL: missing $f"; exit 2; }
done

FREE=$(df -BG --output=avail /mnt/sdc | tail -1 | tr -dc '0-9')
[ "$FREE" -ge 20 ] || { say "FATAL: /mnt/sdc only ${FREE}G free"; exit 2; }
say "/mnt/sdc ${FREE}G free"

# The GEPO loss copy is a verbatim lift of trl's _compute_loss with ONE branch
# changed; it rots silently when trl moves. Prove the diff is still confined before
# spending a GPU-hour on it.
"$PY_TRAIN" "$REPO/scripts/gepo_trainer.py" || { say "FATAL: GEPO loss gate refused"; exit 3; }

# The reward IS the scorer. If the verifier is blind (the daemon=True bug) every
# rollout scores 0.0, the gate looks like it is working, and the run learns nothing.
say "--- reward selftest ---"
"$PY_TRAIN" "$REPO/scripts/lcb_brevity_reward.py" --selftest || {
  say "FATAL: LCB reward selftest failed"; exit 3; }

# ------------------------------------------------------------------ server
serve_up(){ curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

if serve_up; then
  say "vLLM server already up on :$PORT — reusing"
else
  U0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  [ "$U0" -lt 2000 ] || { say "FATAL: GPU0 busy (${U0} MiB); refusing to contend"; exit 2; }
  mkdir -p "$SERVE_DIR"
  # symlink farm without mtp.safetensors: the MTP head is not in the weight index and
  # vLLM has no place to put it.
  for f in "$MODEL"/*; do
    b=$(basename "$f"); [ "$b" = "mtp.safetensors" ] && continue
    ln -sf "$f" "$SERVE_DIR/$b"
  done
  say "launching vLLM on GPU0"
  cd "$REPO" || exit 2
  CUDA_VISIBLE_DEVICES=0 nohup "$PY_SERVE" scripts/vllm_serve_qwen35.py \
      --model "$SERVE_DIR" --port "$PORT" --host 127.0.0.1 \
      --gpu_memory_utilization 0.85 --max_model_len 24576 --dtype bfloat16 \
      --enable_prefix_caching true \
      > "$W/vllm_serve.log" 2>&1 &
  SPID=$!
  disown
  # A sleep is not a readiness predicate. Poll, and fail loudly if the process dies.
  OK=0
  for _ in $(seq 1 120); do
    serve_up && { OK=1; break; }
    kill -0 "$SPID" 2>/dev/null || { say "FATAL: server died, see $W/vllm_serve.log"; exit 3; }
    sleep 15
  done
  [ "$OK" -eq 1 ] || { say "FATAL: server not ready after 30m"; exit 3; }
fi
grep -m1 VLLM_ARCH_REGISTERED "$W/vllm_serve.log" || say "NOTE: arch was already registered"
say "server ready on :$PORT"

# ------------------------------------------------------------------ smoke
run_leg(){
  local name="$1" out="$2"; shift 2
  say "--- $name ---"
  CUDA_VISIBLE_DEVICES=1 "$PY_TRAIN" "$REPO/scripts/gepo_brevity.py" \
      --model "$MODEL" --out "$out" \
      --vllm-base-url "http://127.0.0.1:$PORT" \
      --length-budget "$BUDGET" --length-lambda "$LAMBDA" \
      --max-completion "$MAXCOMP" --num-generations "$G" \
      --beta "$BETA" --lr "$LR" "$@" 2>&1 | tee "$W/${name}.log"
  return "${PIPESTATUS[0]}"
}

if [ "$SKIP_SMOKE" = "0" ]; then
  run_leg smoke "$W/smoke" --limit 4 --grad-accum "$G" --epochs 1 --save-steps 1000
  grep -q GEPO_TRAIN_DONE "$W/smoke.log" || {
    say "FATAL: smoke did not reach GEPO_TRAIN_DONE — not launching the full run"; exit 4; }
  # A smoke that trains on rewards that are all 0.0 proves the plumbing and nothing
  # else. The reward logger prints the realised pass rate; read it back.
  grep -E "LCB_REWARD n=" "$W/smoke.log" | tail -2
  say "SMOKE_OK"
fi

# ------------------------------------------------------------------ full
run_leg full "$W/run1" --grad-accum "$ACCUM" --epochs "$EPOCHS"
RC=$?
grep -q GEPO_TRAIN_DONE "$W/full.log" || { say "FATAL: full run did not finish (rc=$RC)"; exit 5; }

say "adapter at $W/run1"
echo "###### R9_GEPO_DONE $(ts) ######"
