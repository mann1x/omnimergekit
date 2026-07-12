#!/usr/bin/env bash
# rerun_hard77_32k.sh — re-run lcb_hard_77 at the EXTENDED serving config:
#   backend_args.llama_ctx=262144 (256k) + --parallel 8  =>  32768 per-slot
#   generation.max_gen_toks=32768
# All 15 length-caps on the prior 16k run were confirmed GENUINE cut-offs (rep<=0.046,
# code being written at the cut) hitting the OLD per-slot wall (131072/8=16384 - prompt
# ~= 15.4-16.0k). New per-slot 32768 gives ~2x room over the observed caps. Clean
# (finish_reason=stop) rows were KEPT in the sqlite caches; only the deleted length-caps
# + never-run problems regenerate at 32k. Greedy => kept rows are identical at new ctx.
# Pipeline: GPU0 std16 -> cx-c3l4 ; GPU1 coderx -> 128e.
STD16=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF/gemma-4-A4B-98e-v7-coder-it-Q6_K.gguf
CODERX=/mnt/sdc/ml/cx_std16/CX16c4l3-imat-Q6_K.gguf
E128=/mnt/sdc/ml/eval_gguf/128e-Q6_K.gguf
ALTW=/mnt/sdc/ml/cx_altw
set -uo pipefail
PYE=/srv/ml/envs/envs/omnimergekit/bin/python
PYB=/root/anaconda3/envs/omnimergekit/bin/python
export PATH="/srv/ml/envs/envs/omnimergekit/bin:$PATH"
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
RES=/srv/ml/eval_results_lcb_hard77
ts(){ date '+%T %Z'; }
magic_ok(){ [ -f "$1" ] && [ "$(head -c4 "$1" 2>/dev/null)" = "GGUF" ]; }
gpu_free(){ local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '); [ -n "$u" ] && [ "$u" -lt 2000 ]; }
wait_gpu(){ local g=$1 i; for i in $(seq 1 220); do gpu_free "$g" && return 0; sleep 20; done; return 0; }

run_h77(){ # served gpu port q6
  local SN=$1 GPU=$2 PORT=$3 Q6=$4 rc
  magic_ok "$Q6" || { echo "[h77 $(ts)] $SN: Q6 invalid ($Q6); skip"; return 1; }
  echo "[h77 $(ts)] $SN: wait GPU$GPU ..."; wait_gpu "$GPU"
  echo "[h77 $(ts)] $SN: lcb_hard_77 @256k/32k GPU$GPU:$PORT"
  cd /srv/ml/repos/omnimergekit
  CUDA_VISIBLE_DEVICES=$GPU "$PYE" "$OMK" --model "$Q6" --tokenizer "$TOK" --template lcb_hard_77 \
    --backend llama --parallel 8 --port "$PORT" --served-name "$SN" --results-dir "$RES" \
    > "/mnt/sdc/ml/h77_${SN}.log" 2>&1
  rc=$?; echo "[h77 $(ts)] $SN exit=$rc"
}

echo "==================== rerun lcb_hard_77 @256k/32k $(ts) ===================="
( run_h77 std16-h77   0 8440 "$STD16"; \
  run_h77 cx-c3l4-h77 0 8444 "$ALTW/cx-c3l4-imat-Q6_K.gguf" ) &  W0=$!
( run_h77 coderx-h77  1 8441 "$CODERX"; \
  run_h77 128e-h77    1 8445 "$E128" ) &                         W1=$!

# SAT_COLLAPSE guard: once the first servers boot, assert per-slot ctx >= 32768.
( sleep 80
  for sn in std16-h77 coderx-h77; do
    nps=$(grep -oE "n_ctx_per_seq *= *[0-9]+" "/mnt/sdc/ml/h77_${sn}.log" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    nc=$(grep -oE "n_ctx *= *[0-9]+" "/mnt/sdc/ml/h77_${sn}.log" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    if [ -n "$nps" ] && [ "$nps" -lt 32768 ]; then
      echo "[ctx-guard $(ts)] !!! $sn per-slot n_ctx_per_seq=$nps < 32768 — SAT_COLLAPSE RISK (n_ctx=$nc)"
    else
      echo "[ctx-guard $(ts)] OK $sn n_ctx=$nc n_ctx_per_seq=$nps"
    fi
  done ) &

wait "$W0" "$W1"

echo "================ lcb_hard_77 @256k/32k — WEIGHT-SWEEP COHORT $(ts) ================"
"$PYB" - <<PYEOF
import json
RES="$RES/lcb_hard_77"
def res(sn):
    try: return json.load(open(f"{RES}/{sn}/summary.json")).get("score")
    except Exception: return None
rows=[("128e  (unpruned reference)","128e-h77"),
      ("v7-coder STD16  code3/lcb2","std16-h77"),
      ("coderx          code4/lcb3","coderx-h77"),
      ("cx-c3l4         code3/lcb4","cx-c3l4-h77")]
print("%-30s %10s %8s" % ("variant (only --class-weights differ)","hard-77","pass/77"))
for lbl,sn in rows:
    o=res(sn)
    if isinstance(o,(int,float)):
        print("%-30s %9.2f%% %8s" % (lbl, o*100, f"{round(o*77)}/77"))
    else:
        print("%-30s %10s %8s" % (lbl,"NO RESULT","-"))
PYEOF
echo "###### RERUN_HARD77_32K_DONE $(ts) ######"
