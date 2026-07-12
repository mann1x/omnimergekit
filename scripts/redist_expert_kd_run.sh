#!/usr/bin/env bash
# T192 E-ExpertKD runner: KD-train A2's trainable survivor experts to the 128e teacher's
# next-token distribution (logit forward-KL), then loop_screen vs A2 anchors.
# NO --corpus-pad-c4 (English C4 would dilute the multilingual signal); --epochs 100
# cycles the small targeted corpus under the --max-steps cap instead.
# Usage: redist_expert_kd_run.sh <name> <corpus> <train_tensors> <train_layers> <steps> <lr> <gpu> [teacher_load] [seqlen]
set -uo pipefail
NAME="$1"; CORPUS="$2"; TT="$3"; TL="$4"; STEPS="$5"; LR="$6"; GPU="${7:-0}"; TLOAD="${8:-4bit}"; SEQ="${9:-1024}"
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SCR=/srv/ml/scripts
TEACHER=/srv/ml/models/base/gemma-4-26B-A4B-it
STUDENT=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it
WORK=/srv/ml/redist_work
RES=/srv/ml/eval_results_redist
SAMPLE=/mnt/sdc/ml/corpora/loop_screen_sample.jsonl
OUT=/mnt/sdc/ml/google/ekd_${NAME}_62e
mkdir -p "$RES" "$WORK"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L(){ echo "[ekd:$NAME $(date -u +%H:%M:%S)] $*"; }

L "=== [1/3] E-ExpertKD train (tt=$TT layers=$TL steps=$STEPS lr=$LR teacher=$TLOAD seq=$SEQ corpus=$(basename "$CORPUS")) ==="
rm -rf "$OUT"
"$PY" "$SCR/router_kd.py" \
  --base-dir "$TEACHER" --variant-dir "$STUDENT" --out-dir "$OUT" \
  --train-tensors "$TT" --train-layers "$TL" \
  --student-load bf16 --teacher-load "$TLOAD" \
  --teacher-device '{"":0}' --student-device '{"":0}' --gpu-mem-gib 85 \
  --optim paged_adamw8bit --grad-checkpointing \
  --corpus-file "$CORPUS" --epochs 100 --max-samples 100000 \
  --tau 1.0 --lr "$LR" --max-steps "$STEPS" --batch-size 1 --grad-accum 8 \
  --max-seq-len "$SEQ" --no-canary --log-every 10 || { L "KD FAIL"; exit 1; }

L "=== [2/3] loop_screen (200-prompt, greedy bf16, max-new 2048) ==="
"$PY" "$SCR/loop_screen.py" --model "$OUT" --out "$RES/loop_ekd_$NAME.json" --name "ekd_$NAME" \
  --sample "$SAMPLE" --bs 16 --max-new 2048 || { L "SCREEN FAIL"; exit 1; }
"$PY" - "$RES/loop_ekd_$NAME.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); bb = d.get("by_bucket", {})
print("[%s] loop_pct=%s  loops=%s/%s" % (d["name"], d.get("loop_pct"), d.get("loops"), d.get("n")))
print("  by_bucket: " + "  ".join("%s=%s/%s" % (b, v.get("loops"), v.get("n")) for b, v in sorted(bb.items())))
print("  ANCHORS  A2+PES 15.5%% (ML 21/60)   p62 21.5%% (ML 27/60)   128e 0%% (ML 0/60)")
PYEOF

L "=== [3/3] purge merged model (keep result json) ==="
rm -rf "$OUT" && L "purged $OUT"
L "EKD_DONE $NAME"
