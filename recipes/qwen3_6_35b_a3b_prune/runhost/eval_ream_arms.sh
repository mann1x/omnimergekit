#!/usr/bin/env bash
# The R2 table: every REAM arm on ONE basis, WITH the untuned 256e base as a column.
#
# Sampler choice is the load-bearing decision here and it is NOT the project default.
# Qwen3.6 degenerates at temp 0 on *thinking* benches -- the untuned 256e base does it too, so
# it is not a prune artefact -- and the family's tagged cohort is temp 0.6. But temp-0.6
# MultiPL-E has a MEASURED +-5pp run-to-run spread, which is larger than any difference this
# table is trying to resolve; greedy MPE at --parallel 4 has a measured +-0.16pp. Code benches
# run reasoning-OFF, which is exactly the regime where greedy is stable for this family.
# So: greedy @ p4 (template default, no --sampler flag), and the arms are compared only to
# each other and to the base on that one basis.
#
# armD_rpt re-runs an already-measured arm end-to-end. Without it the +-0.16pp band is
# borrowed from a different cohort, and a borrowed band cannot tell a real arm difference
# from jitter on THIS one.
set -uo pipefail

WORK=/mnt/sdc/ream-work
GG=$WORK/gguf
OMK=/srv/ml/repos/omnimergekit
RES=/srv/ml/eval_results/ream_arms
PY=/root/anaconda3/envs/omnimergekit/bin/python
PORT=${PORT:-8097}
GPU=${GPU:-1}
TMPL_FIX=/srv/ml/models/qwen36_chat_template_fixed.jinja
LOG=$WORK/eval_ream_arms.log
CLI=/opt/llama.cpp/build/bin/llama-cli

exec >>"$LOG" 2>&1
echo "=== ream arm eval start $(date -u +%F' '%T) GPU=$GPU PORT=$PORT ==="

[ -s "$TMPL_FIX" ] || { echo "ABORT: fixed chat template missing ($TMPL_FIX)"; exit 1; }
ss -ltn 2>/dev/null | grep -q ":$PORT " && { echo "ABORT: port $PORT busy"; exit 1; }

export HF_HUB_ENABLE_HF_TRANSFER=0 CUDA_VISIBLE_DEVICES="$GPU"
export LM_EVAL_BIN=/root/anaconda3/envs/omnimergekit/bin/lm-eval
export LLAMA_EXTRA="--jinja --chat-template-file $TMPL_FIX"
# Deliberately NOT setting LLAMA_ARG_SPEC_TYPE: the arms' MTP status is not uniform, and a
# speculative-decode path that is on for some columns and off for others is a basis difference.

BASE_G=/srv/ml/models/gguf/Qwen3.6-35B-A3B-256e-GGUF/Qwen3.6-35B-A3B-Q6_K.gguf
BASE_T=/srv/ml/models/Qwen3.6-35B-A3B
PUB_G=/srv/ml/models/gguf/Qwen3.6-35B-A3B-184e-coder-lcbmpe-GGUF/Qwen3.6-35B-A3B-184e-coder-lcbmpe-Q6_K.gguf
PUB_T=/srv/ml/models/Qwen3.6-35B-A3B-184e-coder-lcbmpe

# name | gguf | tokenizer      (order = most decisive first, so a partial run still answers something)
# Every arm column is imat-Q6_K from calibration_datav5.txt -- the same recipe the two anchor
# columns were built with. armC_noimat is the ONE deliberate exception: same weights as
# reamC, quantized without an imatrix, so the pair measures the Q6 imat effect on THIS family
# instead of importing the Gemma-4 crossover number. It is labelled, never averaged in.
COLS=(
  "reamD_ourssal_nomerge|$GG/armD_ourssal_nomerge_imat|$WORK/armD_ourssal_nomerge"
  "reamC_ourssal_merge|$GG/armC_imat|$WORK/armC"
  "reamB_reapsal_merge|$GG/armB_imat|$WORK/armB"
  "reamE_reapsal_nomerge|$GG/armE_imat|$WORK/armE"
  "reamF_rnorm_nomerge|$GG/armF_rnorm_nomerge_imat|$WORK/armF_rnorm_nomerge"
  "base256e_imat|$BASE_G|$BASE_T"
  "pub184e_imat|$PUB_G|$PUB_T"
  "reamD_rpt|$GG/armD_ourssal_nomerge_imat|$WORK/armD_ourssal_nomerge"
  "reamC_noimat_ctrl|$GG/armC_noimat|$WORK/armC"
)
BENCHES=(multipl_e_100 humaneval_full_think)

resolve() {  # dir-or-file -> the Q6_K file
  local p=$1
  if [ -f "$p" ]; then echo "$p"; return 0; fi
  ls "$p"/*-Q6_K.gguf 2>/dev/null | head -1
}

smoke() {  # gguf -> 0 if it emits real text
  local g=$1
  [ -x "$CLI" ] || { echo "     smoke SKIPPED (no llama-cli at $CLI)"; return 0; }
  local out
  out=$("$CLI" -m "$g" -ngl 99 -n 48 --temp 0 -no-cnv \
        -p "def fibonacci(n):" 2>/dev/null | tail -c 400)
  local uniq; uniq=$(echo "$out" | tr -s ' \n' '\n\n' | sort -u | grep -c .)
  echo "     smoke uniq_tokens=$uniq text=[${out:0:120}]"
  [ "${uniq:-0}" -ge 5 ]
}

ran=0; skipped=0; failed=0
for col in "${COLS[@]}"; do
  IFS='|' read -r name src tok <<<"$col"
  g=$(resolve "$src")
  if [ -z "$g" ] || [ ! -s "$g" ]; then
    echo "==== SKIP $name: no Q6_K under $src"; skipped=$((skipped+1)); continue
  fi
  if [ "$(head -c4 "$g")" != "GGUF" ]; then
    echo "==== SKIP $name: $g is not a GGUF"; failed=$((failed+1)); continue
  fi

  echo "==== $name -> $g"
  if ! smoke "$g"; then
    echo "     GGUF SMOKE FAILED -- refusing to spend bench time on a broken quant"
    failed=$((failed+1)); continue
  fi

  for t in "${BENCHES[@]}"; do
    if [ -f "$RES/$t/$name/summary.json" ]; then
      echo "==== SKIP $name / $t (summary exists)"; skipped=$((skipped+1)); continue
    fi
    # Per-bench, and deliberately NOT a blanket override:
    #  * multipl_e_100 freezes max_gen_toks=1024, and the +-0.16pp band was measured at the
    #    template default. Overriding it to 16384 (as the 9-bench suite does) would move the
    #    basis out from under the only band we have. p90 completion is ~275 tok, so 1024 binds
    #    nothing. --parallel 4 is the discriminator's prescribed setting.
    #  * humaneval_full_think wants 16384 gen with a 12288 thinking budget. At --parallel 4 a
    #    49152 ctx gives 12288 per slot -- BELOW the reasoning budget, which is the T172.4
    #    SAT_COLLAPSE bug. Run it at --parallel 2 (24576/slot) like the canonical suite.
    case "$t" in
      multipl_e_100)        par=4 ;;
      humaneval_full_think) par=2 ;;
      *) echo "     no parallel policy for $t -- refusing"; failed=$((failed+1)); continue ;;
    esac
    echo ">>>> $(date -u +%H:%M:%S) START $name / $t (parallel=$par, template-default sampler+gen)"
    "$PY" "$OMK/eval/omk_eval.py" --backend llama --template "$t" --quant q6_k \
      --model "$g" --tokenizer "$tok" --served-name "$name" --port "$PORT" \
      --results-dir "$RES" --parallel "$par" \
      --metadata backend_args.llama_ctx=49152 \
      --metadata backend_args.llama_content_headroom=8192
    echo "<<<< $(date -u +%H:%M:%S) END $name / $t rc=$?"
    [ -f "$RES/$t/$name/summary.json" ] && ran=$((ran+1)) || failed=$((failed+1))
  done
done

echo "=== ream arm eval done $(date -u +%F' '%T) ran=$ran skipped=$skipped failed=$failed ==="
# Read scores from summary.json only -- never from raw results_*.json.
for col in "${COLS[@]}"; do
  IFS='|' read -r name _ _ <<<"$col"
  for t in "${BENCHES[@]}"; do
    s=$RES/$t/$name/summary.json
    [ -f "$s" ] && echo "SCORE $name $t = $("$PY" -c "
import json;d=json.load(open('$s'))
print(round(d.get('score') or 0,4), d.get('metric'), (d.get('sampler') or {}).get('name'))")"
  done
done
echo ">>> REAM_EVAL_DONE ran=$ran failed=$failed"
