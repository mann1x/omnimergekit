#!/usr/bin/env bash
# GPU1: multilingual loop subset (60 prompts) on ml0 + ml1 + v6-coder (bf16, loop_screen)
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1
export HF_XET_HIGH_PERFORMANCE=1
PY=/srv/ml/envs/envs/omnimergekit/bin/python
HF=/srv/ml/envs/envs/omnimergekit/bin/hf
LS=/srv/ml/scripts/loop_screen.py
FULL=/mnt/sdc/ml/corpora/loop_screen_sample.jsonl
SAMP=/mnt/sdc/ml/corpora/loop_screen_multilingual.jsonl
RES=/srv/ml/eval_results_v6_ml_loop
ML0=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-it
ML1=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-ml1-it
V6=/mnt/sdc/ml/google/gemma-4-A4B-98e-v6-coder-it
L(){ echo "[mlloop $(date -u +%H:%M:%S)] $*"; }
mkdir -p "$RES"
"$PY" -c "import json,sys
out=open(sys.argv[2],\"w\")
n=0
for l in open(sys.argv[1]):
    if json.loads(l).get(\"bucket\")==\"multilingual\":
        out.write(l); n+=1
out.close(); print(\"multilingual lines:\",n)" "$FULL" "$SAMP"
# background v6 bf16 download (no GPU)
DLPID=""
if [ ! -f "$V6/config.json" ]; then
  L "starting v6 bf16 download (background)..."
  ( "$HF" download ManniX-ITA/gemma-4-A4B-98e-v6-coder-it --local-dir "$V6" > /srv/ml/logs/v6_bf16_dl.log 2>&1; echo "V6_BF16_DL_DONE rc=$?" >> /srv/ml/logs/v6_bf16_dl.log ) &
  DLPID=$!
fi
run(){ L ">>> loop_screen $2 ($1)"; "$PY" "$LS" --model "$1" --sample "$SAMP" --out "$RES/ml_loop_$2.json" --name "$2" --bs 16 --max-new 2048; L "$2 done"; }
run "$ML0" v7-ml0
run "$ML1" v7-ml1
if [ -n "$DLPID" ]; then L "waiting for v6 bf16 download..."; wait "$DLPID" || true; fi
[ -f "$V6/config.json" ] || { L "FAIL: v6 bf16 missing after download"; tail -5 /srv/ml/logs/v6_bf16_dl.log; exit 1; }
run "$V6" v6coder
L "ML_LOOP_DONE"
"$PY" - "$RES" <<"PYEOF"
import json,sys,glob,os
R=sys.argv[1]
for f in sorted(glob.glob(R+"/ml_loop_*.json")):
    d=json.load(open(f)); b=d.get("by_bucket",{}).get("multilingual",{})
    print("  %-10s loops=%s/%s (%.1f%%)"%(d["name"],d.get("loops"),d.get("n"),d.get("loop_pct")))
PYEOF
