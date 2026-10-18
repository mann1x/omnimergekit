#!/usr/bin/env bash
# pod_v9_toolbench.sh — ON THE POD: tool-eval-bench cell for gemma-4-20B-v9-agentic Q6_K.
# Canonical runner eval/toolbench/run_arm_llamacpp.sh (greedy, --hardmode --weight-by-difficulty,
# seeds 42..46, --jinja, ctx 65536, timeout 600), harness tool-eval-bench v2.6.0 (uv tool).
# Waits until pod_v9_eval_chain.sh has verified the Q6_K, then for the FIRST GPU queue to finish,
# and runs on that GPU so it never shares a card with an omk_eval cell.
set -uo pipefail
W=/workspace
Q6=$W/eval_gguf/gemma-4-20B-v9-agentic-Q6_K.gguf
OUT=$W/eval_results/toolbench
EC=$W/logs/eval_chain.log
LOG=$W/logs/toolbench_chain.log
export PATH="$HOME/.local/bin:$PATH" LLAMA_BIN=$W/llama.cpp/build/bin
L(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
L "toolbench: waiting for the eval chain's Q6_K verification"
until grep -q "Q6_K verified" "$EC" 2>/dev/null; do grep -q "FATAL" "$EC" 2>/dev/null && { L "FATAL eval chain failed"; exit 4; }; sleep 60; done
L "toolbench: waiting for a free GPU queue"
until grep -qE "GPU[01] queue done" "$EC"; do sleep 60; done
G=$(grep -oE "GPU[01] queue done" "$EC" | head -1 | cut -c4)
L "toolbench: GPU$G free -> run_arm_llamacpp.sh v9agentic-q6k (seeds 42..46)"
CUDA_VISIBLE_DEVICES=$G bash $W/omnimergekit/eval/toolbench/run_arm_llamacpp.sh v9agentic-q6k "$Q6" "$OUT" \
  > $W/logs/eval/toolbench.log 2>&1
L "toolbench: rc=$? $(grep -E 'TOOLBENCH_(OK|FAIL|BASIS)' $W/logs/eval/toolbench.log | tr '\n' ' ')"
L "###### TOOLBENCH_DONE ######"
