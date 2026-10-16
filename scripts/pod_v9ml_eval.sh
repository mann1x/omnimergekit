#!/usr/bin/env bash
# v9-ml POD evaluator: one GPU queue of "<bench>:<arm>" cells, each waiting for its arm's BUILD_OK.
# The pod's v9 omk_eval call verbatim (llama backend, parallel 2, frozen greedy template, q8_0 KV,
# /opt/llama.cpp b9700 llama-server). A finished summary.json is never re-run.
# usage: v9ml_pod_eval.sh <gpu> <first_port> <bench:arm>...
set -uo pipefail
G=$1; P=$2; shift 2
W=/workspace; PY=$W/miniconda/envs/omnimergekit/bin/python; OMK=$W/omnimergekit
TOK=$W/gemma-4-20B-v9-agentic; Q=$W/v9ml_quick; RES=$Q/results
export PATH="$W/miniconda/envs/omnimergekit/bin:$PATH" LM_EVAL_BIN="$W/miniconda/envs/omnimergekit/bin/lm_eval"
export HF_ALLOW_CODE_EVAL=1 OMK_GPU_WAIT_S=900
say(){ echo "[eval GPU$G $(date -u +%H:%M:%SZ)] $*"; }
mkdir -p $RES $Q/evallogs
for cell in "$@"; do
  bench=${cell%%:*}; arm=${cell#*:}
  "$PY" - "$OMK/eval/templates/$bench.yaml" <<'PY' || { say "V9ML_EVAL_FAIL template $bench not greedy"; exit 1; }
import sys, yaml
g = (yaml.safe_load(open(sys.argv[1])) or {}).get("generation") or {}
ok = (float(g.get("temperature", -1)) == 0.0 and float(g.get("top_p", -1)) == 1.0
      and int(g.get("top_k", -1)) == 0 and g.get("do_sample") is False)
sys.exit(0 if ok else 1)
PY
  D=$Q/$arm; GG=$D/v9ml-$arm-Q6_K.gguf; SN=v9ml-$arm-q6k; sj=$RES/$bench/$SN/summary.json
  [ -f $sj ] && { say "$bench $arm: done, skip"; continue; }
  until [ -f $D/BUILD_OK ]; do
    grep -q V9ML_BUILD_FAIL $Q/build.log 2>/dev/null && { say "V9ML_EVAL_FAIL builder failed before $arm"; exit 1; }
    sleep 60
  done
  say "$bench $arm: start port $P"
  ( cd $OMK && CUDA_VISIBLE_DEVICES=$G "$PY" eval/omk_eval.py --model $GG --tokenizer $TOK --template $bench \
      --backend llama --parallel 2 --port $P --served-name $SN --results-dir $RES > $Q/evallogs/$bench.$arm.log 2>&1 ); rc=$?
  if [ -f $sj ]; then say "$bench $arm: rc=$rc RESULT $("$PY" -c "import json;d=json.load(open('$sj'));print(d.get('score'),d.get('metric'),(d.get('sampler') or {}).get('name'))")"
  else say "$bench $arm: rc=$rc NO summary.json"; fi
  P=$((P+1)); sleep 30
done
say "QUEUE_DONE"
