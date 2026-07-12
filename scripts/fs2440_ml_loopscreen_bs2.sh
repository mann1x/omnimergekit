#!/usr/bin/env bash
# fs2440 multilingual rumination gate: same 60-prompt ML subset + invocation as the
# 2026-06-03 ml0/ml1/v6 run, so loops are directly comparable. GPU0 (both idle).
set -uo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
LS=/srv/ml/scripts/loop_screen.py
SAMP=/mnt/sdc/ml/corpora/loop_screen_multilingual.jsonl
RES=/srv/ml/eval_results_v6_ml_loop
MODEL=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$RES"
L(){ echo "[fs2440ml $(date -u +%H:%M:%S)] $*"; }
L ">>> loop_screen fs2440 (bf16, GPU0, 60-prompt ML subset)"
"$PY" "$LS" --model "$MODEL" --sample "$SAMP" --out "$RES/ml_loop_fs2440.json" --name fs2440 --bs 16 --max-new 2048
L "fs2440 done"
"$PY" - "$RES/ml_loop_fs2440.json" <<PYEOF
import json,sys
d=json.load(open(sys.argv[1])); bb=d.get("by_bucket",{})
print("[%s] loop_pct=%s loops=%s/%s"%(d["name"],d.get("loop_pct"),d.get("loops"),d.get("n")))
print("  by_bucket: "+"  ".join("%s=%s/%s"%(b,v.get("loops"),v.get("n")) for b,v in sorted(bb.items())))
print("  ANCHORS  v7-ml0 2/60   v7-ml1 2/60   v6-coder 16/60")
PYEOF
echo "FS2440_ML_DONE"
