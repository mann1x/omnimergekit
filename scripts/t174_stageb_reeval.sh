#!/bin/bash
# T174 Stage-B re-eval: HE+ and IFEval only, PATH-fixed (the lm-eval CLI must be
# on PATH for omk_eval's lm-eval subprocess). Q6_K is already built by
# t174_stageb.sh; MultiPL-E already audited CLEAN. Run HE+ on GPU0, IFEval on
# GPU1 in parallel, then re-audit all three vs A2.
set -uo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
OMKBIN=/srv/ml/envs/envs/omnimergekit/bin
export PATH=$OMKBIN:$PATH                 # <-- the fix: lm-eval resolvable
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
AUDIT=/srv/ml/repos/omnimergekit/scripts/audit_full_bench.py
NAME=a2-antiloop-e2
MERGED=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-${NAME}-it
GGUF=$MERGED-GGUF
Q6=$GGUF/$NAME-Q6_K.gguf
RES=/srv/ml/eval_results_tracks_2_3
BASE=a2-62e-fc15_25-p8-s1_0p1_20

command -v lm-eval >/dev/null 2>&1 || { echo "FATAL: lm-eval still not on PATH"; exit 2; }
[ -f "$Q6" ] || { echo "FATAL: Q6_K missing $Q6"; exit 3; }
echo "==================== t174_stageb_reeval $NAME $(date -Iseconds) ===================="
echo "lm-eval -> $(command -v lm-eval)"

# clear the two stale partial dirs (server.log + warmup sqlite, zero samples)
for t in humanevalplus_full ifeval_100; do
  sd=$RES/$t/$NAME
  [ -f "$sd/summary.json" ] || rm -rf "$sd"
done

run_one(){ local gpu=$1 port=$2 tpl=$3
  local sd=$RES/$tpl/$NAME
  [ -f "$sd/summary.json" ] && { echo "[g$gpu] SKIP $tpl (summary exists)"; return 0; }
  pkill -KILL -f "llama-server.*--port $port" 2>/dev/null; sleep 2
  echo "[g$gpu $(date +%H:%M:%S)] eval $tpl"
  CUDA_VISIBLE_DEVICES=$gpu timeout --kill-after=10 --signal=KILL 3000 \
    env PATH="$OMKBIN:$PATH" $PY $OMK \
    --model "$Q6" --tokenizer "$MERGED" --template $tpl --backend llama \
    --port $port --served-name $NAME --results-dir $RES 2>&1 | sed "s/^/[g$gpu] /" | tail -6
  pkill -KILL -f "llama-server.*--port $port" 2>/dev/null; sleep 2
  echo "[g$gpu] DONE $tpl"
}

echo "==================== t174_stageb_reeval EVAL $(date -Iseconds) ===================="
run_one 0 8195 humanevalplus_full & p0=$!
run_one 1 8295 ifeval_100 & p1=$!
wait "$p0" "$p1"

echo "==================== t174_stageb_reeval AUDIT vs $BASE ===================="
for tpl in humanevalplus_full ifeval_100 multipl_e_100; do
  OMK_AUDIT_ROOT=$RES $PY $AUDIT $tpl "$NAME" "$BASE" 2>/dev/null | grep "^AUDIT" || echo "AUDIT_FAIL $tpl"
done
echo "==================== t174_stageb_reeval DONE $(date +%H:%M:%S) ===================="
