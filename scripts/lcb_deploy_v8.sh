#!/usr/bin/env bash
# lcb_deploy_v8.sh [PORT] — v8 (fkbroad-soft2) DEPLOY-sampler LCB-55 + LCB-100 for
# the base-card benchmark row. Runs the SERVED anti-loop sampler vendor_minp_rep
# @ t0.9 (the exact 0/48-gate config) on the ACTUAL v8 ship Q6 GGUF, via the
# env-driven LCB sampler patch (patch_lcb_env_sampler.py) + the *_minprep09
# templates (distinct served-name + sqlite cache, so the deploy cohort never
# merges with the frozen-greedy v8_card run).
#
# Greedy v8_card LCB (87.27 / 90.00) is truncation-noisy because the model
# saturates ~98% of LCB; this measures it in the regime it is actually served.
#
# Sequences AFTER the per-tier sweep: waits for ALL v8_tier_sweep.sh drivers to
# exit (no GPU race with the gating sweep), then grabs the first free GPU
# (<2000 MiB) and runs both benches sequentially. Fully hands-off.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
TPL_DIR=/srv/ml/repos/omnimergekit/eval/templates
MODEL=/mnt/sdc/ml/sft_heal/gemma-4-A4B-98e-v7-coder-fkbroad-soft2-imat-Q6_K.gguf
TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
OUT=/mnt/sdc/ml/v8_lcb_deploy
NAME=v8-fkbroad-soft2-minprep09
# LCB-55 deploy already done (T214): score 0.9636 = 96.36% at
#   /srv/ml/agentic_loop/results/fkbroad_soft2_lcb55_minprep/.../summary.json
# Only the 100-set deploy cell is missing, so run just that.
TPLS=(lcb_medium_100_v4_minprep09)
PORT=${1:-8260}

# vendor_minp_rep @ t0.9 — the 0/48 agentic-gate config (source of truth for LCB sampler)
export LCB_TEMPERATURE=0.9
export LCB_TOP_P=0.95
export LCB_TOP_K=64
export LCB_MIN_P=0.05
export LCB_REPEAT_PENALTY=1.1
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
ts(){ date '+%T %Z'; }

echo "[lcbdep $(ts)] v8 deploy LCB — sampler t=$LCB_TEMPERATURE top_p=$LCB_TOP_P top_k=$LCB_TOP_K min_p=$LCB_MIN_P rep=$LCB_REPEAT_PENALTY"
for f in "$PY" "$OMK" "$MODEL" "$TOK/tokenizer.json"; do
  [ -e "$f" ] || { echo "[lcbdep] FATAL missing $f"; exit 9; }
done
for t in "${TPLS[@]}"; do
  [ -f "$TPL_DIR/${t}.yaml" ] || { echo "[lcbdep] FATAL missing template $t"; exit 9; }
  "$PY" -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" "$TPL_DIR/${t}.yaml" \
    || { echo "[lcbdep] FATAL template $t failed YAML parse"; exit 6; }
done
# confirm the env-driven LCB sampler patch is present (else this would silently run greedy)
grep -q "LCB_TEMPERATURE" /srv/ml/repos/omnimergekit/eval/lcb/lcb_llama_server.py \
  || { echo "[lcbdep] FATAL lcb_llama_server.py not patched for env sampler"; exit 7; }
mkdir -p "$OUT"

# --- wait for the per-tier sweep to fully finish (avoid GPU race) ---
for i in $(seq 1 360); do
  if pgrep -af "v8_tier_sweep.sh" | grep -v "bash -c" | grep -v pgrep >/dev/null 2>&1; then
    echo "[lcbdep $(ts)] tier sweep still running, wait 60s ($i/360)"; sleep 60
  else
    echo "[lcbdep $(ts)] tier sweep done — proceeding"; break
  fi
done

# --- pick first free GPU (<2000 MiB) ---
GPU=""
for i in $(seq 1 240); do
  for g in 0 1; do
    U=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc "0-9")
    if [ "${U:-99999}" -lt 2000 ]; then GPU=$g; break; fi
  done
  [ -n "$GPU" ] && break
  echo "[lcbdep $(ts)] both GPUs busy, wait 30s ($i/240)"; sleep 30
done
[ -n "$GPU" ] || { echo "[lcbdep] FATAL no free GPU"; exit 8; }
echo "[lcbdep $(ts)] using GPU$GPU port=$PORT  model=$(basename "$MODEL")"

for TPL in "${TPLS[@]}"; do
  echo "[lcbdep $(ts)] === $TPL ==="
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$OMK" \
    --model "$MODEL" --template "$TPL" --backend llama --quant gguf \
    --port "$PORT" --results-dir "$OUT" --served-name "$NAME" \
    --tokenizer "$TOK" --parallel 2
  rc=$?
  S="$OUT/$TPL/$NAME/summary.json"
  if [ -f "$S" ]; then
    echo -n "[lcbdep $(ts)] $TPL SCORE="
    "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['score'])" "$S"
  else
    echo "[lcbdep $(ts)] $TPL NO SUMMARY rc=$rc"
  fi
done

echo "###### LCB_DEPLOY_V8_DONE $(ts) ######"
printf "%-30s %8s\n" template score
for TPL in "${TPLS[@]}"; do
  S="$OUT/$TPL/$NAME/summary.json"
  sc=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('score'))" "$S" 2>/dev/null)
  printf "%-30s %8s\n" "$TPL" "${sc:-NA}"
done
