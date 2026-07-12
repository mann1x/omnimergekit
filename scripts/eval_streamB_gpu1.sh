#!/usr/bin/env bash
# Stream B (GPU1, port 8196) — recover the 2 cells contaminated by the 8195 port collision:
#   B1 ml1 HE+164 (server died exit=20)  B2 ml0 LCB-55 (server killed mid-run -> bogus 0.0909)
# Distinct GPU (1) AND distinct port (8196) vs the build LCB on GPU0/8195 -> no collision.
set -uo pipefail
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=1
PORT=8196
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
L(){ echo "[streamB $(date -u +%H:%M:%S)] $*"; }

# purge contaminated/empty subdirs so re-runs are pristine (both are known-invalid)
rm -rf /srv/ml/eval_results_v7coder_ml1/humanevalplus_full/v7coder-ml1-q6k
rm -rf /srv/ml/eval_results_v7coder/lcb_medium_55_v4/v7coder-C6v3lcb-q6k

L "=== B1: ml1 HE+164 (GPU1 port $PORT) ==="
"$PY" "$OMK" --model /mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-ml1-it-GGUF/gemma-4-A4B-98e-v7-coder-ml1-it-Q6_K.gguf \
  --tokenizer /mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-ml1-it \
  --template humanevalplus_full --backend llama --port $PORT \
  --served-name v7coder-ml1-q6k --results-dir /srv/ml/eval_results_v7coder_ml1 2>&1 | tail -15

L "=== B2: ml0 LCB-55 (GPU1 port $PORT) ==="
"$PY" "$OMK" --model /mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-it-GGUF/gemma-4-A4B-98e-v7-coder-it-Q6_K.gguf \
  --tokenizer /mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-it \
  --template lcb_medium_55_v4 --backend llama --port $PORT \
  --served-name v7coder-C6v3lcb-q6k --results-dir /srv/ml/eval_results_v7coder 2>&1 | tail -20

he1=$("$PY" -c "import json;print(json.load(open(\"/srv/ml/eval_results_v7coder_ml1/humanevalplus_full/v7coder-ml1-q6k/summary.json\")).get(\"score\"))" 2>/dev/null)
lcb0=$("$PY" -c "import json;print(json.load(open(\"/srv/ml/eval_results_v7coder/lcb_medium_55_v4/v7coder-C6v3lcb-q6k/summary.json\")).get(\"score\"))" 2>/dev/null)
L "STREAMB_DONE ml1_HEplus=$he1 ml0_LCB=$lcb0"
