#!/usr/bin/env bash
# Split-half stability of the t106 LIFT pin selection.
# 71 positions touching ~20 of 128 experts per layer is either real concentration or
# sampling noise. Two disjoint halves through the identical pipeline, scored against the
# SHARED background arm (the population reference, 1.95M positions — not the thin side).
# Read the result against the chance line, not against 100%.
set -u
cd /mnt/sdc/harness
P=/srv/ml/envs/envs/omnimergekit/bin/python3.11
R=/srv/ml/repos/omnimergekit/scripts/run_agentic_eog_map.sh
S=/srv/ml/repos/omnimergekit/scripts
T=/mnt/sdc/agentbench/tierb
O=/mnt/sdc/ml/google

$P "$S/split_eog_corpus.py" "$T/eog_corpus_v4_t106.jsonl" \
   "$T/eog_t106_half_A.jsonl" "$T/eog_t106_half_B.jsonl"
for H in A B; do
  echo "########## HALF $H ##########"
  bash "$R" 0 "$T/eog_t106_half_$H.jsonl" "$O/expert_neuron_eog_v4_t106_half$H.json" \
       "<turn|>" 106 model
  echo "HALF${H}_RC=$?"
done
echo "########## SPLIT-HALF OVERLAP ##########"
$P "$S/eog_map_overlap.py" \
   --half-a "$O/expert_neuron_eog_v4_t106_halfA.json" \
   --half-b "$O/expert_neuron_eog_v4_t106_halfB.json" \
   --bg-map "$O/expert_neuron_eog_v4_t106_bg.json"
echo "EOG_SPLIT_DONE"
