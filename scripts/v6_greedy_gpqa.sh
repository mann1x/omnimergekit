#!/usr/bin/env bash
# v6 GREEDY GPQA head-to-head: published Q6_K (GPU0) vs AC-imatrix Q6_K (GPU1).
#
# WHY GREEDY: the existing sampled pair (T=0.6) put 25.3% of the bench in
# discordant flips, giving a 95% CI of [-12.05,+1.95] pp -- unable to separate a
# healthy imatrix from a 10pp regression. Greedy removes the sampler variance so
# the paired comparison can actually resolve the -5.05pp.
#
# This is a SEPARATE COHORT from the published `recommended` cells. Never tabulate
# these rows against sampled rows: summary.json.sampler.name will read
# template_default here and recommended there.
set -uo pipefail
WAIT_PID="${WAIT_PID:-3536264}"
D=/mnt/sdc/ml/omnimerge-v6
TOK=$D/tokenizer
R=/srv/ml/eval_results/v6_greedy_gpqa
OMK=/srv/ml/repos/omnimergekit
PYBIN=/srv/ml/envs/envs/omnimergekit/bin/python3.11
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH
ts(){ date -u +"[%H:%M:%SZ]"; }

command -v lm-eval >/dev/null || { echo "$(ts) ABORT: lm-eval not on PATH"; exit 3; }
for f in "$D/Qwen3.8-27B-Omnimerge-v6-Q6_K.gguf" "$D/Qwen3.8-27B-Omnimerge-v6-Q6_K-AC.gguf"; do
  [ -s "$f" ] || { echo "$(ts) ABORT: missing $f"; exit 2; }
done
echo "$(ts) both GGUFs present"

echo "$(ts) waiting for running eval pid $WAIT_PID to finish"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "$(ts) pid $WAIT_PID gone -- GPUs released"
sleep 30

launch(){  # $1=gpu $2=gguf $3=name $4=port
  echo "$(ts) launching $3 on GPU$1 port$4"
  CUDA_VISIBLE_DEVICES=$1 nohup "$PYBIN" "$OMK/eval/omk_eval.py" \
    --backend llama --template gpqa_diamond_full --quant q6_k \
    --model "$2" --tokenizer "$TOK" --served-name "$3" \
    --port "$4" --results-dir "$R/$3" \
    > "/srv/ml/logs/v6_greedy_$3.log" 2>&1 &
  echo "  pid=$!"
}

launch 0 "$D/Qwen3.8-27B-Omnimerge-v6-Q6_K.gguf"    v6pub_q6k_greedy 8095
sleep 45
launch 1 "$D/Qwen3.8-27B-Omnimerge-v6-Q6_K-AC.gguf" v6ac_q6k_greedy  8096

sleep 120
echo "$(ts) sampler verification (MUST read template_default on both)"
grep -ho "OMK_SAMPLER.*" /srv/ml/logs/v6_greedy_v6pub_q6k_greedy.log /srv/ml/logs/v6_greedy_v6ac_q6k_greedy.log 2>/dev/null | cut -c1-120
wait
echo "$(ts) V6_GREEDY_GPQA_BOTH_DONE"
