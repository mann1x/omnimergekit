#!/usr/bin/env bash
# v9-ml Tier-A map: map A basis (2048 tok, answer channel, fixed template, 128e) + two new
# held-out categories (generic_multilingual, generic_ifeval). No Tier-B: the 43-session
# replay is reused from expert_neuron_v9_agentic_code.json; the 5 recomputed generic_*
# categories are the same-basis check before the merge.
set -u
export CUDA_VISIBLE_DEVICES=0
cd /mnt/sdc/harness
PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11
S=/srv/ml/repos/omnimergekit/scripts
C=/mnt/sdc/ml/corpora/v9ml
OUT=/mnt/sdc/ml/google/expert_neuron_v9ml_tiera.json
echo "[v9ml] start $(date -u +%FT%TZ) gpu=$CUDA_VISIBLE_DEVICES analysis_sha=$(sha256sum $S/expert_neuron_analysis_v5_targeted.py | cut -c1-16)"
PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" "$S/expert_neuron_analysis_v5_targeted.py" \
  --variant code \
  --model /mnt/sdc/ml/models/gemma-4-26B-A4B-it \
  --gpu-budget-gib 90 \
  --tier-a-max-tokens 2048 \
  --no-tier-a-thinking \
  --window-tokens 2048 --window-overlap 0 \
  --chat-template-file /mnt/sdc/v7rework/gemma4_a4b_fixed_chat_template.jinja \
  --tier-a-extra multilingual=$C/multilingual_calib_v9ml.jsonl \
  --tier-a-extra ifeval=$C/ifeval_calib_v9ml.jsonl \
  --out "$OUT"
rc=$?
echo "[v9ml] V9ML_MAP_RC=$rc $(date -u +%FT%TZ)"
