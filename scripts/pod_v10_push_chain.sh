#!/usr/bin/env bash
# pod_v10_push_chain.sh — ON THE POD: wait for pod_v10_chain.sh's V10_CHAIN_DONE, then push both v10 Q4_K_M
# builds as private test tags of mannix/gemma-4-20B-v9-agentic: <map>-Q4_K_M + vision-/mtp-/mtp-vision- (no "v10" in
# model or tag names — v10 is only the recipe, owner 2026-10-17).
# Owner 2026-10-17: "push them as private ollama test tags in https://ollama.com/mannix/gemma-4-20B-v9-agentic".
set -uo pipefail
W=/workspace; PY=$W/miniconda/envs/omnimergekit/bin/python; S=$W/omnimergekit/scripts; L=$W/v10/logs/chain.log
say(){ echo "[push $(date -u +%FT%TZ)] $*"; }
for _ in $(seq 1 1440); do
  grep -q V10_CHAIN_DONE $L 2>/dev/null && break
  grep -qE "V10_FAIL|CHAIN_FAIL" $L 2>/dev/null && { say "PUSH_FAIL build chain failed — nothing pushed"; exit 1; }
  tmux has-session -t v10chain 2>/dev/null || { say "PUSH_FAIL v10chain gone without V10_CHAIN_DONE"; exit 1; }
  sleep 30
done
grep -q V10_CHAIN_DONE $L || { say "PUSH_FAIL chain not done after 12 h"; exit 1; }
for n in hub_k12 hub_k9; do
  "$PY" $S/pod_v10_ollama_test_tags.py --gguf $W/v10/gguf/$n/gemma-4-20B-v9-agentic-$n-Q4_K_M.gguf --prefix $n \
    || { say "PUSH_FAIL $n"; exit 1; }
done
say "PUSH_CHAIN_DONE"
