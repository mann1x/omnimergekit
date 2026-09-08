#!/usr/bin/env bash
# THE DECIDER: LCB-v6-77q (all-hard) on every surviving arm + freshly-run anchors.
#
# WHY NEW ANCHOR COLUMNS INSTEAD OF REUSING THE JULY ONES
# -------------------------------------------------------
# qwen_suite/lcb_v6_77q already has base256e / pub184e / qwencoder. They are NOT reusable here:
#   1. sampler = "recommended" (temp 0.6 / top_p 0.95 / top_k 20, do_sample=true). Every R2
#      cell is greedy (template_default). A greedy row and a sampled row must never share a
#      table -- and at temp 0.6 an arm-vs-arm gap of ~1pp is unreadable anyway.
#   2. they ran 2026-07-13; neither their server.log nor their summary.json records a llama.cpp
#      build, so binary identity with today's runs cannot be established from the artifacts.
# Unverifiable basis + wrong sampler = re-run. At ~30-45 min/column that is cheap.
#
# GREEDY IS THE RISK, AND IT IS GATED. Qwen3.6 degenerates at temp 0 on THINKING benches, and
# unlike MPE/HE+ this template runs reasoning ON (enable_thinking + 12288 budget). So pub184e
# runs FIRST and its score is checked: a catastrophic value means greedy is unusable on this
# bench for this family, and the chain stops rather than burning 5 more hours proving it seven
# times. The July recommended-sampler value for the same weights is 0.5844, so anything above
# the floor below is a plausible greedy reading, not a degeneration.
#
# --parallel 8 is the template's own prescription: llama_ctx=262144 / 8 = 32768 per slot,
# which is >= max_gen_toks and far above the 12288 thinking budget (no T172.4 SAT_COLLAPSE).
#
# GPU1 only. Exported, not polled.
set -uo pipefail

WORK=/mnt/sdc/ream-work
GG=$WORK/gguf
OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
RES=/srv/ml/eval_results/ream_arms
TMPL_FIX=/srv/ml/models/qwen36_chat_template_fixed.jinja
LOG=$WORK/chain_lcb.log
PORT=${PORT:-8099}
FLOOR=${FLOOR:-0.30}          # below this on the first column => greedy degeneration, stop

export CUDA_VISIBLE_DEVICES=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export LM_EVAL_BIN=/root/anaconda3/envs/omnimergekit/bin/lm-eval
export LLAMA_EXTRA="--jinja --chat-template-file $TMPL_FIX"

exec >>"$LOG" 2>&1
say() { echo ">>> [$(date -u +%H:%M:%S)] $*"; }
echo "=== chain_lcb start $(date -u +%F' '%T) ==="
[ -s "$TMPL_FIX" ] || { echo "ABORT: fixed chat template missing"; exit 2; }

BASE_G=/srv/ml/models/gguf/Qwen3.6-35B-A3B-256e-GGUF/Qwen3.6-35B-A3B-Q6_K.gguf
BASE_T=/srv/ml/models/Qwen3.6-35B-A3B
PUB_G=/srv/ml/models/gguf/Qwen3.6-35B-A3B-184e-coder-lcbmpe-GGUF/Qwen3.6-35B-A3B-184e-coder-lcbmpe-Q6_K.gguf
PUB_T=/srv/ml/models/Qwen3.6-35B-A3B-184e-coder-lcbmpe

# name | gguf-or-dir | tokenizer
# Order = decisiveness. pub184e first because it is the greedy sanity gate AND the column the
# HF replies quote. armD second: bit-identical weights to pub184e, independently quantized, so
# the pair measures build jitter ON THIS BENCH instead of importing the MPE band.
COLS=(
  "lcb_pub184e|$PUB_G|$PUB_T"
  "lcb_armD_ourssal_nomerge|$GG/armD_ourssal_nomerge_imat|$WORK/armD_ourssal_nomerge"
  "lcb_base256e|$BASE_G|$BASE_T"
  "lcb_armB_reapsal_merge|$GG/armB_imat|$WORK/armB"
  "lcb_armE_reapsal_nomerge|$GG/armE_imat|$WORK/armE"
  "lcb_armF_rnorm_nomerge|$GG/armF_rnorm_nomerge_imat|$WORK/armF_rnorm_nomerge"
  "lcb_armG_merge_gs2|$GG/armG_imat|$WORK/armG"
)

resolve() { local p=$1; if [ -f "$p" ]; then echo "$p"; else ls "$p"/*-Q6_K.gguf 2>/dev/null | head -1; fi; }
score_of() { "$OMKPY" -c "
import json,sys
try: print(json.load(open(sys.argv[1])).get('score'))
except Exception: print('')" "$1" 2>/dev/null; }

ran=0; failed=0; first=1
for col in "${COLS[@]}"; do
  IFS='|' read -r name src tok <<<"$col"
  g=$(resolve "$src")
  if [ -z "$g" ] || [ ! -s "$g" ]; then say "SKIP $name: no Q6_K under $src"; failed=$((failed+1)); continue; fi
  s=$RES/lcb_v6_77q/$name/summary.json
  if [ -f "$s" ]; then say "SKIP $name (summary exists)"; continue; fi

  say ">>>> START $name (greedy, template default, parallel=8)"
  "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template lcb_v6_77q --quant q6_k \
      --model "$g" --tokenizer "$tok" --served-name "$name" --port "$PORT" \
      --results-dir "$RES" --parallel 8
  say "<<<< END $name rc=$?"

  if [ -f "$s" ]; then
    ran=$((ran+1))
    v=$(score_of "$s"); say "SCORE $name = $v"
    # Greedy degeneration gate -- only on the first column, and only as a STOP, never a rescore.
    if [ "$first" -eq 1 ]; then
      bad=$("$OMKPY" -c "
import sys
try: sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)
except Exception: sys.exit(1)" "$v" "$FLOOR"; echo $?)
      if [ "$bad" = "0" ]; then
        say "ABORT: $name greedy score $v < floor $FLOOR -- greedy looks degenerate on this"
        say "       thinking bench. Re-run this cohort with --sampler-profile qwen3.6-35B-A3B"
        say "       --sampler recommended as a TAGGED cohort. Not continuing."
        echo ">>> LCB_CHAIN_ABORT_GREEDY_DEGENERATE score=$v"
        exit 4
      fi
      say "greedy sanity OK ($v >= $FLOOR) -- continuing with the remaining columns"
      first=0
    fi
  else
    failed=$((failed+1)); say "FAIL $name: no summary.json"
  fi
done

say "=== chain_lcb done ran=$ran failed=$failed ==="
"$OMKPY" "$WORK/lcb_readout.py" $(for c in "${COLS[@]}"; do echo "ream_arms/${c%%|*}"; done) 2>&1 || true
echo ">>> LCB_CHAIN_DONE ran=$ran failed=$failed"
