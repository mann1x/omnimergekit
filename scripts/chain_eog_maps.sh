#!/usr/bin/env bash
# Build the two agentic_eog maps that have never existed:
#   t106 = the DELIVERY close (agent stops calling tools and answers) -- the terminator
#          that bears on non-convergence. 71 model-owned positions from manic v4, which
#          the continuation test licensed (only 1/81 closes lead the teacher back to a
#          tool call).
#   t49  = the TOOL-CALL close. 573 model-owned positions from the SWE qset_A traces.
#          Never profiled by any shipped map; v7-coder pinned id 50, an OPENER.
# Both run with --emit-role model. For 49 that is a no-op (573 of 573 raw are model-owned);
# for 106 it excludes 66.7% prompt-side contamination.
set -u
R=/srv/ml/repos/omnimergekit/scripts/run_agentic_eog_map.sh
T=/mnt/sdc/agentbench/tierb
O=/mnt/sdc/ml/google

echo "########## MAP 1/2: t106 delivery close (v4, 71 positions) ##########"
bash "$R" 0 "$T/eog_corpus_v4_t106.jsonl" "$O/expert_neuron_eog_v4_t106.json" "<turn|>" 106 model
echo "MAP1_RC=$?"

echo "########## MAP 2/2: t49 tool-call close (SWE qset_A, 573 positions) ##########"
bash "$R" 0 "$T/eog_corpus_sweA_t49.jsonl" "$O/expert_neuron_eog_sweA_t49.json" "<tool_call|>" 49 model
echo "MAP2_RC=$?"
echo "EOG_MAPS_CHAIN_DONE"
