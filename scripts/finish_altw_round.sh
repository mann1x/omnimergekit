#!/usr/bin/env bash
# finish_altw_round.sh — finish the alt-weight round robustly after the rescue
# raced a mid-write Q6. cx-c3l4 HE+ failed (loaded a half-written GGUF); its Q6
# is valid now. cx-c35l25 HE+ is running on GPU0 via build_cx_c35l25.sh.
#   Phase 1: re-run cx-c3l4 HE+ on GPU1 (valid Q6).
#   Phase 2: wait cx-c3l4 + cx-c35l25 HE+ summaries -> print 4-way HE+ table.
#   Phase 3: lcb_v6_55 on all 3 variants (GPU1: c3l4->c3l3 ; GPU0: c35l25).
#   Phase 4: print v6-55 table (overall + medium/hard) incl. anchors.
# All GPU gates check the GGUF MAGIC, not mere file existence (the rescue bug).
set -uo pipefail
PYE=/srv/ml/envs/envs/omnimergekit/bin/python
PYB=/root/anaconda3/envs/omnimergekit/bin/python
export PATH="/srv/ml/envs/envs/omnimergekit/bin:$PATH"
export HF_ALLOW_CODE_EVAL=1
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
WORK=/mnt/sdc/ml/cx_altw
RES_HE=/srv/ml/eval_results_cx_altw
RESV6=/srv/ml/eval_results_lcb_v6
ts(){ date '+%T %Z'; }
magic_ok(){ [ -f "$1" ] && [ "$(head -c4 "$1" 2>/dev/null)" = "GGUF" ]; }
gpu_free(){ local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '); [ -n "$u" ] && [ "$u" -lt 2000 ]; }
wait_q6(){ local f=$1 i; for i in $(seq 1 160); do magic_ok "$f" && return 0; sleep 20; done; return 1; }
wait_gpu(){ local g=$1 i; for i in $(seq 1 150); do gpu_free "$g" && return 0; sleep 20; done; return 0; }
he_summary(){ find "$RES_HE" -path "*$1*" -name summary.json 2>/dev/null | grep -i humaneval | head -1; }

run_he(){ # tag gpu port
  local TAG=$1 GPU=$2 PORT=$3 Q6=$WORK/$1-imat-Q6_K.gguf rc
  echo "[he $(ts)] $TAG: wait valid Q6 ..."; wait_q6 "$Q6" || { echo "[he $(ts)] $TAG Q6 never valid"; return 1; }
  echo "[he $(ts)] $TAG: wait GPU$GPU ..."; wait_gpu "$GPU"
  echo "[he $(ts)] $TAG: HE+ GPU$GPU:$PORT"
  CUDA_VISIBLE_DEVICES=$GPU "$PYE" "$OMK" --model "$Q6" --tokenizer "$TOK" --template humanevalplus_full \
    --backend llama --port "$PORT" --served-name "$TAG" --results-dir "$RES_HE" > "$WORK/${TAG}_heplus.log" 2>&1
  rc=$?; echo "[he $(ts)] $TAG HE+ exit=$rc"
}

run_v6(){ # tag gpu port
  local TAG=$1 GPU=$2 PORT=$3 Q6=$WORK/$1-imat-Q6_K.gguf SN=$1-v6 i rc
  echo "[v6 $(ts)] $TAG: wait valid Q6 + HE+ done ..."
  wait_q6 "$Q6" || { echo "[v6 $(ts)] $TAG Q6 invalid; skip"; return 1; }
  for i in $(seq 1 160); do [ -n "$(he_summary "$TAG")" ] && break; sleep 30; done
  echo "[v6 $(ts)] $TAG: wait GPU$GPU ..."; wait_gpu "$GPU"
  echo "[v6 $(ts)] $TAG: lcb_v6_55 GPU$GPU:$PORT"
  cd /srv/ml/repos/omnimergekit
  CUDA_VISIBLE_DEVICES=$GPU "$PYE" "$OMK" --model "$Q6" --tokenizer "$TOK" --template lcb_v6_55 \
    --backend llama --port "$PORT" --served-name "$SN" --results-dir "$RESV6" > "$WORK/${TAG}_v6_55.log" 2>&1
  rc=$?; echo "[v6 $(ts)] $TAG v6-55 exit=$rc"
}

echo "==================== finish alt-weight round $(ts) ===================="
# Phase 1: re-run cx-c3l4 HE+ on GPU1 (cx-c35l25 HE+ already on GPU0 via build script)
run_he cx-c3l4 1 8421

# Phase 2: wait for both pending HE+ summaries, then table
echo "[round $(ts)] waiting cx-c3l4 + cx-c35l25 HE+ summaries ..."
for i in $(seq 1 120); do
  [ -n "$(he_summary cx-c3l4)" ] && [ -n "$(he_summary cx-c35l25)" ] && break; sleep 30
done
echo "================ ALT-WEIGHT HE+ TABLE $(ts) ================"
"$PYB" - <<PYEOF
import json, glob
RES="$RES_HE"
def sc(tag):
    for c in glob.glob(f"{RES}/**/summary.json", recursive=True):
        if f"/{tag}/" in c and "humanevalplus" in c:
            try: return json.load(open(c)).get("score")
            except Exception: pass
    return None
rows=[("STD16  code3/lcb2 (published)",0.9329),("coderx code4/lcb3",0.9329),
      ("cx-c35l25 code3.5/lcb2.5", sc("cx-c35l25")),
      ("cx-c3l3 code3/lcb3", sc("cx-c3l3")),
      ("cx-c3l4 code3/lcb4", sc("cx-c3l4"))]
print("%-32s %s" % ("variant (only --class-weights differs)","HE+ imat-Q6 greedy"))
for k,v in rows: print("%-32s %s" % (k,(f"{round(v*100,2)}%") if isinstance(v,(int,float)) else "NO RESULT"))
PYEOF

# Phase 3: v6-55 on all 3 (GPU1: c3l4 -> c3l3 ; GPU0: c35l25)
( run_v6 cx-c3l4 1 8431; run_v6 cx-c3l3 1 8432 ) &  W1=$!
( run_v6 cx-c35l25 0 8430 ) &                       W0=$!
wait "$W1" "$W0"

echo "================ ALT-WEIGHT lcb_v6_55 TABLE $(ts) ================"
"$PYB" - <<PYEOF
import json, sys
sys.path.insert(0,"/srv/ml/repos/omnimergekit/eval/lcb")
from lcb_helpers import load_lcb
RESV6="$RESV6/lcb_v6_55"
ids=json.load(open("/srv/ml/repos/omnimergekit/eval/lcb/lcb_v6_55_taskids.json"))
dmap={p["task_id"]:p.get("difficulty","?") for p in load_lcb(limit=999, task_ids=ids)}
def overall(sn):
    try: return json.load(open(f"{RESV6}/{sn}/summary.json")).get("score")
    except Exception: return None
def medhard(sn):
    try: d=json.load(open(f"{RESV6}/{sn}/lcb_result.json"))
    except Exception: return None
    items=None
    for k in ("problems","per_problem","results","details"):
        if isinstance(d.get(k),list): items=d[k]; break
    if items is None and isinstance(d,list): items=d
    mp=mt=hp=ht=0
    for it in items or []:
        if not isinstance(it,dict): continue
        tid=it.get("task_id") or it.get("id"); ok=it.get("passed")
        if ok is None: ok=it.get("pass")
        df=dmap.get(str(tid))
        if df=="medium": mt+=1; mp+=int(bool(ok))
        elif df=="hard": ht+=1; hp+=int(bool(ok))
    return (mp,mt,hp,ht)
rows=[("v7-coder STD16 code3/lcb2","v7coder-q6"),("coderx code4/lcb3","cx16-c4l3"),
      ("cx-c35l25 code3.5/lcb2.5","cx-c35l25-v6"),("cx-c3l3 code3/lcb3","cx-c3l3-v6"),
      ("cx-c3l4 code3/lcb4","cx-c3l4-v6")]
print("%-28s %9s %8s %7s" % ("variant","overall","medium","hard"))
for lbl,sn in rows:
    o=overall(sn); mh=medhard(sn)
    print("%-28s %9s %8s %7s" % (lbl,(f"{round(o*100,2)}%") if isinstance(o,(int,float)) else "NO RESULT",
          (f"{mh[0]}/{mh[1]}" if mh and mh[1] else "?"),(f"{mh[2]}/{mh[3]}" if mh and mh[3] else "?")))
PYEOF
echo "###### FINISH_ALTW_ROUND_DONE $(ts) ######"
