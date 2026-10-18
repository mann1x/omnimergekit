#!/usr/bin/env bash
# pod_v9_ifeval_sampler.sh <sampler> <gpu> <port> — IFEval-100 on the v9-ml ml_k8 Q6_K with a named sampler from
# eval/models/gemma-4-v9-agentic.yaml. The k8 suite's omk_eval call verbatim except --sampler (suite: greedy_pen = 96.00).
# Writes IFEVAL_DONE / IFEVAL_FAIL to /workspace/v9samp/samplertest.log.
set -uo pipefail
S=$1; G=$2; PORT=$3
W=/workspace; PY=$W/miniconda/envs/omnimergekit/bin/python; OMK=$W/omnimergekit; OUT=$W/v9samp
LOG=$OUT/samplertest.log; mkdir -p $OUT/results $OUT/logs
say(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOG; }
Q6=$W/gguf/gemma-4-20B-v9ml-ml_k8/gemma-4-20B-v9ml-ml_k8-Q6_K.gguf; TOK=$W/v9pub/gemma-4-20B-v9-agentic
[ "$(sha256sum $Q6 | cut -c1-16)" = 5957b63a9b0733cf ] || { say "IFEVAL_FAIL $S Q6_K sha"; exit 1; }
export PATH="$W/miniconda/envs/omnimergekit/bin:$PATH" LM_EVAL_BIN="$W/miniconda/envs/omnimergekit/bin/lm_eval" HF_ALLOW_CODE_EVAL=1
SN=v9ml-k8-imat-q6k-${S#greedy_}
say "ifeval $S start GPU$G port $PORT served $SN"
( cd $OMK && CUDA_VISIBLE_DEVICES=$G "$PY" eval/omk_eval.py --model $Q6 --tokenizer $TOK --template ifeval_100 \
    --backend llama --parallel 2 --port $PORT --served-name $SN --sampler $S \
    --results-dir $OUT/results > $OUT/logs/ifeval_100.$S.log 2>&1 ); rc=$?
sj=$OUT/results/ifeval_100/$SN/summary.json
if [ -f $sj ]; then say "IFEVAL_DONE $S rc=$rc $("$PY" -c "import json;d=json.load(open('$sj'));s=d['sampler'];print(d['score'],d['metric'],s['name'],s['resolved'])")"
else say "IFEVAL_FAIL $S rc=$rc no summary.json"; fi
