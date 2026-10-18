#!/usr/bin/env bash
# pod_v9_samplertest.sh — owner sampler test 2026-10-16: temp 0.8 / repeat_penalty 1.07 / presence_penalty 0.03
#   GPU0: IFEval-100, GREEDY + the candidate penalties (sampler greedy_pen107: rep 1.07 / last 96 / presence 0.03),
#         the k8 suite's Q6_K + call verbatim except the sampler (suite cell: greedy_pen 1.06 / 0.025 -> 96.00)
#   GPU1: tool-eval-bench cohort64k basis, 5 seeds, the published Q4_K_M, stock llama.cpp b9700, no drafter,
#         candidate sampler sent PER REQUEST (temp 0.8 / top_p 0.95 / top_k 64 / min_p 0 / rep 1.07 / last 96 / presence 0.03)
set -uo pipefail
W=/workspace; PY=$W/miniconda/envs/omnimergekit/bin/python; OMK=$W/omnimergekit; T=$W/v9pub/tb
OUT=$W/v9samp; LOG=$OUT/samplertest.log; mkdir -p $OUT/results $OUT/logs
say(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOG; }
die(){ say "SAMPTEST_FAIL $*"; exit 1; }
Q6=$W/gguf/gemma-4-20B-v9ml-ml_k8/gemma-4-20B-v9ml-ml_k8-Q6_K.gguf
Q4=$W/v9pub/eval_gguf/gemma-4-20B-v9-agentic-Q4_K_M.gguf
Q4SIDE=$W/v9pub/gguf/gemma-4-20B-v9-agentic/gemma-4-20B-v9-agentic-Q4_K_M.gguf.sha256
TOK=$W/v9pub/gemma-4-20B-v9-agentic
[ "$(sha256sum $OMK/eval/models/gemma-4-v9-agentic.yaml | cut -c1-16)" = 05041cf92dd0d1db ] || die "profile sha"
[ "$(sha256sum $OMK/eval/toolbench/run_cohort.sh | cut -c1-16)" = fa7aa1080f8d7e90 ] || die "run_cohort sha"
[ -x $T/bin/llama-server-cohort ] || die "no stock llama-server wrapper"
[ "$(sha256sum $Q4 | cut -d' ' -f1)" = "$(cut -d' ' -f1 $Q4SIDE)" ] || die "Q4_K_M sha != sidecar"
Q6SHA=$(sha256sum $Q6 | cut -d' ' -f1) || die "Q6_K sha"
say "Q6_K $Q6SHA  Q4_K_M $(cut -c1-16 $Q4SIDE)"
"$PY" - "$OMK/eval/templates/ifeval_100.yaml" <<'PYC' || die "ifeval template not greedy"
import sys, yaml
g = (yaml.safe_load(open(sys.argv[1])) or {}).get("generation") or {}
sys.exit(0 if (float(g.get("temperature", -1)) == 0.0 and g.get("do_sample") is False) else 1)
PYC
export PATH="$W/miniconda/envs/omnimergekit/bin:$PATH" LM_EVAL_BIN="$W/miniconda/envs/omnimergekit/bin/lm_eval" HF_ALLOW_CODE_EVAL=1

( SN=v9ml-k8-imat-q6k-pen107
  cd $OMK && CUDA_VISIBLE_DEVICES=0 "$PY" eval/omk_eval.py --model $Q6 --tokenizer $TOK --template ifeval_100 \
     --backend llama --parallel 2 --port 8810 --served-name $SN --sampler greedy_pen107 \
     --results-dir $OUT/results > $OUT/logs/ifeval_100.pen107.log 2>&1; rc=$?
  sj=$OUT/results/ifeval_100/$SN/summary.json
  if [ -f $sj ]; then say "IFEVAL_DONE rc=$rc $("$PY" -c "import json;d=json.load(open('$sj'));s=d['sampler'];print(d['score'],d['metric'],s['name'],s['resolved'])")"
  else say "IFEVAL_FAIL rc=$rc no summary.json"; fi ) &
IFP=$!

TBO=$OUT/cohort64k_v9ml_cand107; mkdir -p $TBO
printf '%s\n' "v9-agentic (ml_k8) arm, CANDIDATE sampler test, cohort64k basis, $(date -u +%F)" \
  "model    ManniX-ITA/gemma-4-20B-v9-agentic-GGUF gemma-4-20B-v9-agentic-Q4_K_M.gguf sha256 $(cut -d' ' -f1 $Q4SIDE)" \
  "drafter  none" \
  "sampler  owner candidate, sent PER REQUEST: temp 0.8 / top_p 0.95 / top_k 64 / min_p 0 / repeat_penalty 1.07 / repeat_last_n 96 / presence_penalty 0.03" \
  "server   STOCK llama.cpp b9700 (9724f664e) llama-server, vast pod 55947563 RTX 3090 GPU1" \
  "harness  tool-eval-bench 2.6.0 (992a6978), --hardmode --weight-by-difficulty, ctx 65536, pressure 0.25, timeout 600, seeds 42..46" \
  "compare  cohort64k v9-agentic (same build/host/server): greedy + 1.06/96/0.025 = 137.6 +-5.4" > $TBO/RUN_BASIS.txt
say "toolbench: 5 seeds on GPU1"
export PATH=$HOME/.local/bin:$PATH
CUDA_VISIBLE_DEVICES=1 OMK_TB_BIN=$T/bin/llama-server-cohort OMK_TB_MODELS=$T/models OMK_TB_OUT=$TBO OMK_TB_PORT=8267 \
OMK_TB_PNG=$TBO/arm.png \
OMK_TB_MODELS_LIST="v9-agentic|gemma-4-20B-v9-agentic-Q4_K_M.gguf||temp=0.8,top_p=0.95,top_k=64,min_p=0,presence_penalty=0.03,repeat_penalty=1.07,repeat_last_n=96" \
  bash $OMK/eval/toolbench/run_cohort.sh >> $OUT/logs/run_cohort.cand107.out 2>&1; trc=$?
n=$(grep -rlE "\*\*Total Points\*\*" $TBO/v9-agentic-s4[2-6] 2>/dev/null | wc -l)
[ "$n" -eq 5 ] && say "TB_DONE 5/5 rc=$trc" || say "TB_FAIL only $n/5 rc=$trc"
wait $IFP
say "SAMPTEST_END"
