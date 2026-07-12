#!/usr/bin/env bash
# rescore_lcb_hard_77.sh — rescore the weight-sweep cohort on the new all-HARD
# lcb_hard_77 bench (the discriminating successor to the saturated v6-55).
# Self-gates on FINISH_ALTW_ROUND_DONE so it never fights the round for GPUs.
# Cohort = 4 (dropped cx-c35l25 + cx-c3l3 — both HE+ 91.46 losers with nothing to
# recommend them). Kept: 128e (unpruned reference) + STD16 + coderx + cx-c3l4
# (code3/lcb4 held the 93.29 HE+ anchor AND is the aggressive LCB lean).
# 2-GPU pipelined:  GPU0: std16 -> cx-c3l4      GPU1: coderx -> 128e
# Q6 GGUFs (imat-Q6, all same 98e recipe, only --class-weights differ):
STD16=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF/gemma-4-A4B-98e-v7-coder-it-Q6_K.gguf
CODERX=/mnt/sdc/ml/cx_std16/CX16c4l3-imat-Q6_K.gguf
E128=/mnt/sdc/ml/eval_gguf/128e-Q6_K.gguf      # unpruned 128-expert reference anchor
ALTW=/mnt/sdc/ml/cx_altw
set -uo pipefail
PYE=/srv/ml/envs/envs/omnimergekit/bin/python
PYB=/root/anaconda3/envs/omnimergekit/bin/python
export PATH="/srv/ml/envs/envs/omnimergekit/bin:$PATH"
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
RES=/srv/ml/eval_results_lcb_hard77
GATE=/mnt/sdc/ml/finish_altw_round.log
ts(){ date '+%T %Z'; }
magic_ok(){ [ -f "$1" ] && [ "$(head -c4 "$1" 2>/dev/null)" = "GGUF" ]; }
gpu_free(){ local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '); [ -n "$u" ] && [ "$u" -lt 2000 ]; }
wait_gpu(){ local g=$1 i; for i in $(seq 1 200); do gpu_free "$g" && return 0; sleep 20; done; return 0; }
mkdir -p "$RES"

run_h77(){ # served gpu port q6
  local SN=$1 GPU=$2 PORT=$3 Q6=$4 rc
  magic_ok "$Q6" || { echo "[h77 $(ts)] $SN: Q6 invalid ($Q6); skip"; return 1; }
  echo "[h77 $(ts)] $SN: wait GPU$GPU ..."; wait_gpu "$GPU"
  echo "[h77 $(ts)] $SN: lcb_hard_77 GPU$GPU:$PORT"
  cd /srv/ml/repos/omnimergekit
  CUDA_VISIBLE_DEVICES=$GPU "$PYE" "$OMK" --model "$Q6" --tokenizer "$TOK" --template lcb_hard_77 \
    --backend llama --port "$PORT" --served-name "$SN" --results-dir "$RES" > "/mnt/sdc/ml/h77_${SN}.log" 2>&1
  rc=$?; echo "[h77 $(ts)] $SN exit=$rc"
}

echo "==================== rescore lcb_hard_77 (FAST-TRACK) $(ts) ===================="
# GPU0 worker: start NOW — GPU0 is free (cx-c35l25 killed). No round gate.
( run_h77 std16-h77   0 8440 "$STD16"; \
  run_h77 cx-c3l4-h77 0 8444 "$ALTW/cx-c3l4-imat-Q6_K.gguf" ) &  W0=$!
# GPU1 worker: GATE on FINISH_ALTW_ROUND_DONE — cx-c3l4 v6-55 + cx-c3l3 guard must
# clear GPU1 before coderx-h77 starts, else they collide on GPU1.
( echo "[gpu1-gate $(ts)] waiting FINISH_ALTW_ROUND_DONE before GPU1 hard-77 ..."
  for i in $(seq 1 240); do grep -q "FINISH_ALTW_ROUND_DONE" "$GATE" 2>/dev/null && break; sleep 30; done
  echo "[gpu1-gate $(ts)] round wrapped; GPU1 hard-77 starting"
  run_h77 coderx-h77  1 8441 "$CODERX"; \
  run_h77 128e-h77    1 8445 "$E128" ) &                         W1=$!
wait "$W0" "$W1"

echo "================ lcb_hard_77 — WEIGHT-SWEEP COHORT $(ts) ================"
"$PYB" - <<PYEOF
import json, glob
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
echo "###### RESCORE_LCB_HARD77_DONE $(ts) ######"
