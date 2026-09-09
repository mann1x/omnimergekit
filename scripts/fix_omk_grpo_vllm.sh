#!/usr/bin/env bash
# TRL 1.12.0 declares vllm<=0.27.1,>=0.19.0. Installing vllm==0.29.0 alongside it
# type-checks at pip level (no extra requested) but breaks at import:
#   ImportError: cannot import name NCCLTrainerSendWeightsArgs
#     from vllm.distributed.weight_transfer.nccl_engine
# vLLM 0.29.0 renamed that symbol. Pin to 0.27.1 = newest vLLM TRL 1.12.0 supports.
set -euo pipefail
E=/srv/ml/envs/envs/omk-grpo
"$E/bin/pip" install "trl[vllm]==1.12.0" "vllm==0.27.1"
echo "=== resolved after pin ==="
"$E/bin/python" - <<PY
import importlib.metadata as md
for p in ["torch","vllm","trl","transformers","peft","accelerate","datasets","bitsandbytes"]:
    try: print(f"{p:16s} {md.version(p)}")
    except Exception as e: print(f"{p:16s} MISSING")
PY
"$E/bin/python" /srv/ml/scripts/probe_grpo_env.py
echo "FIX_OK"
