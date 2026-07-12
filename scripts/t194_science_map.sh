#!/bin/bash
# T194: build v7-science map (targeted_gpqa) via 128e Tier-B replay (reuse base_v7
# Tier-A), then merge targeted_gpqa into v7-code -> combined v7-code+gpqa map.
set -uo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
G=/mnt/sdc/ml/google
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
MLCALIB=/mnt/sdc/ml/corpora/multilingual_calib.jsonl
BASE_MAP=$G/expert_neuron_base_v7.json
TIERB=/mnt/sdc/ml/corpora/v5_gpqa_science_traces.json
SCI_MAP=$G/expert_neuron_v7_science.json
CODE_MAP=$G/expert_neuron_v7_code.json
COMBINED=$G/expert_neuron_v7_code_gpqa.json
LOG=/srv/ml/logs/t194_science_map.log
mkdir -p /srv/ml/logs
exec > >(tee -a "$LOG") 2>&1
L(){ echo "[t194 $(date -u +%H:%M:%S)] $*"; }

L "=== preflight ==="
for f in "$TIERB" "$BASE_MAP" "$MLCALIB" "$CODE_MAP"; do
  [ -f "$f" ] || { L "FATAL missing $f"; exit 2; }
done
[ -d "$SRC" ] || { L "FATAL missing $SRC"; exit 2; }

L "=== [1/2] producer --variant science (Tier-B replay, reuse base_v7 Tier-A) ==="
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  $PY "$SCR/expert_neuron_analysis_v5_targeted.py" \
    --variant science --model "$SRC" --gpu-budget-gib 90 \
    --tier-b-json "$TIERB" --load-tier-a-from "$BASE_MAP" \
    --multilingual-calib "$MLCALIB" --no-tier-a-thinking \
    --out "$SCI_MAP" || { L "PRODUCER FAIL"; exit 1; }

L "=== verify science map has targeted_gpqa ==="
$PY -c "
import json
d=json.load(open('$SCI_MAP')); c=d.get('categories',{})
assert 'targeted_gpqa' in c, 'no targeted_gpqa: '+str(list(c.keys()))
print('science map cats:', list(c.keys()))
" || { L "VERIFY FAIL"; exit 1; }

L "=== [2/2] merge targeted_gpqa into v7-code -> combined ==="
$PY - <<PYEOF
import json
code=json.load(open("$CODE_MAP")); sci=json.load(open("$SCI_MAP"))
cm,sm=code["metadata"],sci["metadata"]
for k in ("num_layers","num_experts","intermediate_size"):
    assert int(cm[k])==int(sm[k]), f"dim mismatch {k}"
code["categories"]["targeted_gpqa"]=sci["categories"]["targeted_gpqa"]
code.setdefault("metadata",{}).setdefault("merged_categories",[]).append(
    {"from":"$SCI_MAP","category":"targeted_gpqa"})
json.dump(code, open("$COMBINED","w"))
print("combined cats:", list(code["categories"].keys()))
PYEOF
L "T194_SCIENCE_MAP_DONE -> $COMBINED"
