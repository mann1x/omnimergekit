#!/usr/bin/env bash
# v9-ml quick test EVALUATOR (bs2 GPU1): the pod's v9 omk_eval call (llama backend, parallel 2, frozen greedy
# template, omk_eval's q8_0 KV) on each arm's no-imatrix Q6_K. ifeval_v9fail10 ONLY (user 2026-10-15: decide on ifeval_100 after). Runs each arm as it
# lands; pass 2: ifeval_100 on every arm. A finished summary.json is never re-run.
set -uo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11
OMK=/mnt/sdc/ml/v9ml/omk
TOK=/mnt/sdc/ml/models/gemma-4-20B-v9-agentic
Q=/mnt/sdc/ml/v9ml/quick
RES=$Q/results
ARMS=(A shared_k2 shared_k4 shared_k8 ml_k8 if_k8 shared_k6)
export LLAMA_BIN=/mnt/sdc/llama.cpp-311d4211bf/build/bin LM_EVAL_BIN=/srv/ml/envs/envs/omnimergekit/bin/lm_eval
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH HF_ALLOW_CODE_EVAL=1 OMK_GPU_WAIT_S=900
P=8620
say(){ echo "[eval $(date -u +%H:%M:%SZ)] $*"; }
for _T in ifeval_v9fail10; do
"$PY" - "$OMK/eval/templates/$_T.yaml" <<'PY' || { echo "V9ML_EVAL_FAIL template not greedy"; exit 1; }
import sys, yaml
g = (yaml.safe_load(open(sys.argv[1])) or {}).get("generation") or {}
ok = (float(g.get("temperature", -1)) == 0.0 and float(g.get("top_p", -1)) == 1.0
      and int(g.get("top_k", -1)) == 0 and g.get("do_sample") is False)
print(("gate OK greedy: " if ok else "NOT GREEDY: ") + sys.argv[1]); sys.exit(0 if ok else 1)
PY
done
mkdir -p $RES $Q/evallogs
for bench in ifeval_v9fail10; do
  for arm in "${ARMS[@]}"; do
    D=$Q/$arm; G=$D/v9ml-$arm-Q6_K.gguf; SN=v9ml-$arm-q6k; sj=$RES/$bench/$SN/summary.json
    [ -f $sj ] && { say "$bench $arm: done, skip"; continue; }
    until [ -f $D/BUILD_OK ]; do
      grep -q V9ML_BUILD_FAIL $Q/build.log 2>/dev/null && { say "V9ML_EVAL_FAIL builder failed before $arm"; exit 1; }
      sleep 60
    done
    say "$bench $arm: start port $P"
    ( cd $OMK && CUDA_VISIBLE_DEVICES=1 "$PY" eval/omk_eval.py --model $G --tokenizer $TOK --template $bench \
        --backend llama --parallel 2 --port $P --served-name $SN --results-dir $RES > $Q/evallogs/$bench.$arm.log 2>&1 ); rc=$?
    if [ -f $sj ]; then say "$bench $arm: rc=$rc RESULT $("$PY" -c "import json;d=json.load(open('$sj'));print(d.get('score'),d.get('metric'),(d.get('sampler') or {}).get('name'))")"
    else say "$bench $arm: rc=$rc NO summary.json"; fi
    P=$((P+1)); sleep 30
  done
  say "PASS_DONE $bench"
done
say "V9ML_EVAL_ALL_DONE"
