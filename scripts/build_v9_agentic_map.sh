#!/usr/bin/env bash
# v9 agentic competence map — the v7-coder recipe with its two measured defects fixed.
#
# What v7-coder's map actually was (from expert_neuron_v7_code.json metadata):
#   tier_a_max_tokens 128   -> reaches <turn|> only 3/6 of the time (bug-360); at 2048, 6/6
#   tier_b_source v5_code_pass_traces.json -> 360 SINGLE-TURN HumanEval/LCB completions,
#                                             prompt/completion pairs, ZERO tool-bearing
#                                             messages. The map never saw a tool call.
#
# This run changes exactly those two axes and holds everything else at the shipped recipe
# (variant code, window 2048/overlap 0, thinking OFF, same base, same fixed template), so
# a diff of the resulting drop maps attributes cleanly.
#
#   tier A: --tier-a-max-tokens 2048   (the generation actually terminates)
#   tier B: qset_FULL.json  -- 43 agentic sessions, ALL multi-turn (38-285 messages),
#           2275 assistant messages carrying tool calls, 2267 with reasoning,
#           7 languages (java/rust/javascript/ruby/go/php/cpp_c), all resolved=True
set -u
cd /mnt/sdc/harness
PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11
S=/srv/ml/repos/omnimergekit/scripts
OUT=/mnt/sdc/ml/google/expert_neuron_v9_agentic_code.json

echo "[v9map] start $(date -u +%H:%M:%S)Z"
CUDA_VISIBLE_DEVICES=0 "$PY" "$S/expert_neuron_analysis_v5_targeted.py" \
  --variant code \
  --model /srv/ml/models/base/gemma-4-26B-A4B-it \
  --gpu-budget-gib 90 \
  --tier-a-max-tokens 2048 \
  --no-tier-a-thinking \
  --tier-b-json /mnt/sdc/agentbench/tierb/qset_FULL.json \
  --window-tokens 2048 --window-overlap 0 \
  --chat-template-file /mnt/sdc/v7rework/gemma4_a4b_fixed_chat_template.jinja \
  --out "$OUT"
echo "V9MAP_RC=$?"
echo "[v9map] done $(date -u +%H:%M:%S)Z"
"$PY" - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); m = d.get("metadata", {})
print("[v9map] categories:", list(d.get("categories", {})))
for k in ("tier_a_max_tokens", "tier_a_thinking", "tier_b_source", "tier_b_trace_count",
          "tier_b_shape", "tier_b_set_counts", "chat_template_file"):
    print("   %-22s %s" % (k, m.get(k)))
PY
echo "V9MAP_CHAIN_DONE"
