#!/usr/bin/env bash
# rescue_cx_altw_heplus.sh — the orchestrator's heplus() died on a set-u unbound-var
# bug, so HE+ never auto-launched. cx-c3l3 HE+ was launched manually (GPU0:8420).
# This handles cx-c3l4: wait for the orchestrator to build its imat-Q6, wait GPU1
# free (coderx v6-55 finishing), run cx-c3l4 HE+ on GPU1, then report all 4.
set -uo pipefail
PYE=/srv/ml/envs/envs/omnimergekit/bin/python
PYB=/root/anaconda3/envs/omnimergekit/bin/python
export PATH="/srv/ml/envs/envs/omnimergekit/bin:$PATH"
export HF_ALLOW_CODE_EVAL=1
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
WORK=/mnt/sdc/ml/cx_altw
RES=/srv/ml/eval_results_cx_altw
Q6B=$WORK/cx-c3l4-imat-Q6_K.gguf
ts(){ date '+%T %Z'; }

echo "[rescue $(ts)] waiting for cx-c3l4 imat-Q6 (orchestrator building) ..."
for _ in $(seq 1 200); do [ -f "$Q6B" ] && break; sleep 30; done
[ -f "$Q6B" ] || { echo "[rescue $(ts)] cx-c3l4 Q6 never appeared; abort"; exit 1; }
echo "[rescue $(ts)] cx-c3l4 Q6 ready; waiting GPU1 free ..."
for _ in $(seq 1 90); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')
  [ -n "$u" ] && [ "$u" -lt 2000 ] && break; sleep 30
done
echo "[rescue $(ts)] launch cx-c3l4 HE+ GPU1:8421"
CUDA_VISIBLE_DEVICES=1 "$PYE" "$OMK" --model "$Q6B" --tokenizer "$TOK" --template humanevalplus_full \
  --backend llama --port 8421 --served-name cx-c3l4 --results-dir "$RES" \
  > "$WORK/cx-c3l4_heplus.log" 2>&1 || echo "[rescue $(ts)] WARN cx-c3l4 HE+ rc=$?"

echo "[rescue $(ts)] waiting for both HE+ summaries ..."
for _ in $(seq 1 90); do
  c3=$(find "$RES" -path "*cx-c3l3*" -name summary.json 2>/dev/null | grep -i humaneval | head -1)
  c4=$(find "$RES" -path "*cx-c3l4*" -name summary.json 2>/dev/null | grep -i humaneval | head -1)
  [ -n "$c3" ] && [ -n "$c4" ] && break; sleep 30
done

echo "================ ALT-WEIGHTS HE+ RESULT $(ts) ================"
"$PYB" - <<PYEOF
import json, glob
RES="$RES"
def sc(tag):
    for c in glob.glob(f"{RES}/**/summary.json", recursive=True):
        if f"/{tag}/" in c and "humaneval" in c.lower():
            try: return json.load(open(c)).get("score")
            except Exception: pass
    return None
rows=[("STD16  code3/lcb2 (published)", 0.9329),
      ("coderx code4/lcb3", 0.9329),
      ("cx-c3l3 code3/lcb3 (new)", sc("cx-c3l3")),
      ("cx-c3l4 code3/lcb4 (new)", sc("cx-c3l4"))]
print("%-32s %s" % ("variant","HE+ (imat-Q6, greedy)"))
for k,v in rows:
    print("%-32s %s" % (k, (f"{round(v*100,2)}%") if isinstance(v,(int,float)) else "NO RESULT"))
PYEOF
echo "###### CX_ALTW_RESCUE_DONE $(ts) ######"
