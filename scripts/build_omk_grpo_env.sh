#!/usr/bin/env bash
# Build the dedicated GRPO training env on bs2.
# Separate from `omnimergekit` (trl 1.9.0, torch 2.10+cu128) and from `vllm`
# (serving-only, pinned, DO NOT MODIFY). This one owns TRL 1.12.0 + a vLLM new
# enough to colocate generation inside the trainer.
set -euo pipefail

CONDA=/srv/ml/envs/bin/conda
ENV_NAME=omk-grpo
ENV_DIR=/srv/ml/envs/envs/$ENV_NAME
PY_VER=3.11

# --- root-fs floor gate (bs2 rule: never go below 200G free on /) -----------
FREE_G=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
if [ "$FREE_G" -lt 215 ]; then
  echo "ABORT: / has ${FREE_G}G free; need >=215G headroom to build a ~10G env above the 200G floor" >&2
  exit 3
fi
echo "[gate] root fs free=${FREE_G}G  OK"

if [ -d "$ENV_DIR" ]; then
  echo "ABORT: $ENV_DIR already exists — refusing to clobber. Remove it explicitly first." >&2
  exit 4
fi

$CONDA create -y -n "$ENV_NAME" "python=$PY_VER"
PIP="$ENV_DIR/bin/pip"
"$PIP" install --upgrade pip setuptools wheel

# Top-level intent pins. Everything else is a transitive resolution and gets
# captured into the lockfile afterwards.
"$PIP" install \
  "vllm==0.29.0" \
  "trl==1.12.0" \
  "peft" \
  "accelerate" \
  "datasets" \
  "bitsandbytes" \
  "math-verify"

echo "=== resolved core versions ==="
"$ENV_DIR/bin/python" - <<PY
import importlib.metadata as md
for p in ["torch","vllm","trl","transformers","peft","accelerate","datasets","bitsandbytes","numpy"]:
    try: print(f"{p:16s} {md.version(p)}")
    except Exception as e: print(f"{p:16s} MISSING ({e})")
import torch
print("torch.cuda:", torch.version.cuda, "| devices:", torch.cuda.device_count())
print("sm caps:", [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])
PY

echo "=== import smoke: trl GRPO + vllm ==="
"$ENV_DIR/bin/python" -c "
from trl import GRPOTrainer, GRPOConfig
import vllm, trl, inspect
print(trl, trl.__version__, vllm, vllm.__version__)
sig = inspect.signature(GRPOConfig.__init__)
for k in [loss_type,importance_sampling_level,scale_rewards,vllm_mode,vllm_tensor_parallel_size,epsilon,beta,steps_per_generation]:
    print(f GRPOConfig.{k:28s}, PRESENT if k in sig.parameters else *** ABSENT ***)
"
echo "BUILD_OK"
