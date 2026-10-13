#!/usr/bin/env bash
# Background arms + lift ranking + split-half stability.
#
# The emit maps alone cannot be ranked: raw emit-position mass is topped by globally-hot
# experts. These arms accumulate the COMPLEMENT (same corpus, same role filter, next token
# is NOT the terminator) to give lift = emit share / background share its denominator.
set -u
cd /mnt/sdc/harness
P=/srv/ml/envs/envs/omnimergekit/bin/python3.11
R=/srv/ml/repos/omnimergekit/scripts/run_agentic_eog_map.sh
S=/srv/ml/repos/omnimergekit/scripts
T=/mnt/sdc/agentbench/tierb
O=/mnt/sdc/ml/google

echo "########## BG 1/2: t106 background (v4) ##########"
EMIT_MODE=background bash "$R" 0 "$T/eog_corpus_v4_t106.jsonl" \
  "$O/expert_neuron_eog_v4_t106_bg.json" "<turn|>" 106 model
echo "BG1_RC=$?"

echo "########## BG 2/2: t49 background (SWE qset_A) ##########"
EMIT_MODE=background bash "$R" 0 "$T/eog_corpus_sweA_t49.jsonl" \
  "$O/expert_neuron_eog_sweA_t49_bg.json" "<tool_call|>" 49 model
echo "BG2_RC=$?"

echo "########## LIFT: t106 ##########"
$P "$S/rank_agentic_eog_lift.py" --eog-map "$O/expert_neuron_eog_v4_t106.json" \
   --bg-map "$O/expert_neuron_eog_v4_t106_bg.json" --topk 4 \
   --out "$O/eog_v4_t106_lift_pins.txt"
echo "########## LIFT: t49 ##########"
$P "$S/rank_agentic_eog_lift.py" --eog-map "$O/expert_neuron_eog_sweA_t49.json" \
   --bg-map "$O/expert_neuron_eog_sweA_t49_bg.json" --topk 4 \
   --out "$O/eog_sweA_t49_lift_pins.txt"
echo "EOG_BG_CHAIN_DONE"
