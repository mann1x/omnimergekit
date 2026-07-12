#!/usr/bin/env bash
# queue_altw_v6_55.sh — run lcb_v6_55 (discriminating LCB) on the 3 alt-weight
# variants after each one's HE+ frees its GPU. 2 fixed-GPU workers, pipelined:
#   GPU1: cx-c3l4 -> cx-c3l3   ·   GPU0: cx-c35l25
# Each run gates on (its HE+ summary exists => HE+ done => GPU free) before
# launching, so v6-55 never collides with the in-flight HE+/build on that GPU.
# Final table: overall + medium/hard for all 3 + the v7-coder/coderx anchors.
set -uo pipefail
PYE=/srv/ml/envs/envs/omnimergekit/bin/python
PYB=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
WORK=/mnt/sdc/ml/cx_altw
RES_HE=/srv/ml/eval_results_cx_altw     # HE+ results (gate)
RESV6=/srv/ml/eval_results_lcb_v6       # v6-55 results
ts(){ date '+%T %Z'; }

run_v6() {   # $1=tag $2=gpu $3=port
  local TAG=$1 GPU=$2 PORT=$3 Q6=$WORK/${TAG}-imat-Q6_K.gguf SN=${TAG}-v6
  echo "[v6q $(ts)] $TAG: waiting HE+ done (summary) ..."
  for _ in $(seq 1 200); do
    find "$RES_HE" -path "*${TAG}*" -name summary.json 2>/dev/null | grep -qi humaneval && break
    sleep 30
  done
  [ -f "$Q6" ] || { echo "[v6q $(ts)] $TAG: Q6 missing, skip"; return 1; }
  echo "[v6q $(ts)] $TAG: waiting GPU$GPU free ..."
  for _ in $(seq 1 150); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" -lt 2000 ] && break; sleep 20
  done
  echo "[v6q $(ts)] $TAG: lcb_v6_55 GPU$GPU:$PORT"
  cd /srv/ml/repos/omnimergekit
  CUDA_VISIBLE_DEVICES=$GPU env PATH="/srv/ml/envs/envs/omnimergekit/bin:$PATH" "$PYE" "$OMK" \
    --model "$Q6" --tokenizer "$TOK" --template lcb_v6_55 --backend llama --port "$PORT" \
    --served-name "$SN" --results-dir "$RESV6" > "$WORK/${TAG}_v6_55.log" 2>&1 || echo "[v6q $(ts)] WARN $TAG rc=$?"
}

echo "==================== alt-weights lcb_v6_55 queue $(ts) ===================="
( run_v6 cx-c3l4 1 8431; run_v6 cx-c3l3 1 8432 ) &  W1=$!
( run_v6 cx-c35l25 0 8430 ) &                       W0=$!
wait "$W1" "$W0"

echo "================ lcb_v6_55 — alt-weight sweep $(ts) ================"
"$PYB" - <<PYEOF
import json, glob, sys
sys.path.insert(0,"/srv/ml/repos/omnimergekit/eval/lcb")
RESV6="$RESV6/lcb_v6_55"
ids=json.load(open("/srv/ml/repos/omnimergekit/eval/lcb/lcb_v6_55_taskids.json"))
from lcb_helpers import load_lcb
dmap={p["task_id"]:p.get("difficulty","?") for p in load_lcb(limit=999, task_ids=ids)}
def overall(sn):
    f=f"{RESV6}/{sn}/summary.json"
    try: return json.load(open(f)).get("score")
    except Exception: return None
def medhard(sn):
    rj=f"{RESV6}/{sn}/lcb_result.json"
    try: d=json.load(open(rj))
    except Exception: return None
    items=None
    for k in ("problems","per_problem","results","details"):
        if isinstance(d.get(k),list): items=d[k]; break
    if items is None and isinstance(d,list): items=d
    mp=mt=hp=ht=0
    for it in items or []:
        if not isinstance(it,dict): continue
        tid=it.get("task_id") or it.get("id"); ok=it.get("passed");
        if ok is None: ok=it.get("pass")
        diff=dmap.get(str(tid))
        if diff=="medium": mt+=1; mp+=int(bool(ok))
        elif diff=="hard": ht+=1; hp+=int(bool(ok))
    return (mp,mt,hp,ht)
rows=[("v7-coder STD16 (code3/lcb2)","v7coder-q6"),
      ("coderx code4/lcb3","cx16-c4l3"),
      ("cx-c35l25 code3.5/lcb2.5","cx-c35l25-v6"),
      ("cx-c3l3 code3/lcb3","cx-c3l3-v6"),
      ("cx-c3l4 code3/lcb4","cx-c3l4-v6")]
print("%-30s %9s %9s %7s" % ("variant","overall","medium","hard"))
for lbl,sn in rows:
    o=overall(sn); mh=medhard(sn)
    os_=(f"{round(o*100,2)}%") if isinstance(o,(int,float)) else "NO RESULT"
    md=f"{mh[0]}/{mh[1]}" if mh and mh[1] else "?"
    hd=f"{mh[2]}/{mh[3]}" if mh and mh[3] else "?"
    print("%-30s %9s %9s %7s" % (lbl, os_, md, hd))
PYEOF
echo "###### ALTW_V6_55_QUEUE_DONE $(ts) ######"
