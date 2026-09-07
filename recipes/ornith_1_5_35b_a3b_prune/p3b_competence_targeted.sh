#!/usr/bin/env bash
# P3b — competence map WITH the two targeted channels.
# The first map (competence_ornith35b.json) carried ONLY the 8 prompt-only router
# categories, so corpus_targeted_lcb / corpus_targeted_mpe did not exist and the coder
# drop map was built on substituted prompt-only categories (bug-673) and is VOID.
# This rebuild concatenates the balanced corpus with the teacher's PASS-only full-CoT
# solutions on the DISJOINT LCB (103q) and MultiPL-E tail (175q) sets.
# Writes a NEW output; the original map is never overwritten.
set -uo pipefail
ts(){ date -u +%H:%M:%S; }
PY=/workspace/venv-omk/bin/python
R=/workspace/ornith_prune/results
FULL=$R/router_calib_corpus_ornith_full.jsonl
cd /workspace/ornith_prune || { echo "no ornith_prune dir"; exit 3; }

cat "$R/router_calib_corpus_ornith.jsonl" \
    "$R/router_calib_corpus_lcb_ornith.jsonl" \
    "$R/router_calib_corpus_mpe_ornith.jsonl" > "$FULL"

echo "[$(ts)] merged corpus census:"
"$PY" /workspace/ornith_prune/_corpus_census.py "$FULL" || { echo "[$(ts)] GATE FAILED - not launching"; exit 2; }

echo "[$(ts)] P3b competence map (bf16 on A100 80GB)"
nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader
"$PY" expert_neuron_analysis_v5_targeted.py \
    --model /workspace/models/Ornith-1.5-35B-A3B \
    --device cuda:0 \
    --corpus "$FULL" \
    --corpus-cat-field bench \
    --tc-only \
    --checkpoint-every 25 \
    --out "$R/competence_ornith35b_targeted.json"
rc=$?
echo "[$(ts)] competence rc=$rc"
ls -la "$R/competence_ornith35b_targeted.json" 2>/dev/null
echo "[$(ts)] P3B_COMPETENCE_DONE"
