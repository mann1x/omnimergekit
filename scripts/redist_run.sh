#!/usr/bin/env bash
# T191 closed-form redistribution experiment: capture(driver corpus) -> fit/emit a
# recovered 62e -> loop_screen vs A2 anchors. The capture corpus MUST be disjoint
# from the loop_screen sample (or the fold is fit on the test set).
# Usage: redist_run.sh <driver> <method> <calib_corpus> [gpu] [max_seqs]
set -uo pipefail
DRIVER="$1"; METHOD="$2"; CORPUS="$3"
GPU="${4:-0}"; MAXSEQ="${5:-120}"
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SCR=/srv/ml/scripts
TEACHER=/srv/ml/models/base/gemma-4-26B-A4B-it
STUDENT=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it
KM=/srv/ml/scripts/a2_keep_metadata.json
SAMPLE=/mnt/sdc/ml/corpora/loop_screen_sample.jsonl
WORK=/srv/ml/redist_work
RES=/srv/ml/eval_results_redist
NAME="${DRIVER}_${METHOD}"
OUT=/mnt/sdc/ml/google/redist_${NAME}_62e
mkdir -p "$RES" "$WORK"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L(){ echo "[$NAME $(date -u +%H:%M:%S)] $*"; }

L "=== [1/4] capture (driver=$DRIVER corpus=$(basename "$CORPUS") seqs=$MAXSEQ) ==="
"$PY" "$SCR/redist.py" capture --driver "$DRIVER" --method "$METHOD" --teacher "$TEACHER" \
  --corpus "$CORPUS" --keep-meta "$KM" --max-seqs "$MAXSEQ" --max-tokens 512 \
  --device cuda:0 --workdir "$WORK" --scripts-dir "$SCR" || { L "CAPTURE FAIL"; exit 1; }
CAP="$WORK/capture_${DRIVER}_${METHOD}.pt"

L "=== [2/4] fit + emit ($METHOD -> recovered 62e) ==="
rm -rf "$OUT"
"$PY" "$SCR/redist.py" redistribute --method "$METHOD" --driver "$DRIVER" --capture "$CAP" \
  --teacher "$TEACHER" --student "$STUDENT" --keep-meta "$KM" --emit "$OUT" \
  --device cuda:0 --scripts-dir "$SCR" || { L "FIT/EMIT FAIL"; exit 1; }

L "=== [3/4] loop_screen (200-prompt, greedy bf16, max-new 2048) ==="
"$PY" "$SCR/loop_screen.py" --model "$OUT" --out "$RES/loop_$NAME.json" --name "$NAME" \
  --sample "$SAMPLE" --bs 16 --max-new 2048 || { L "SCREEN FAIL"; exit 1; }
"$PY" - "$RES/loop_$NAME.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); bb = d.get("by_bucket", {})
print("[%s] loop_pct=%s  loops=%s/%s" % (d["name"], d.get("loop_pct"), d.get("loops"), d.get("n")))
print("  by_bucket: " + "  ".join("%s=%s/%s" % (b, v.get("loops"), v.get("n")) for b, v in sorted(bb.items())))
print("  ANCHORS  A2+PES: 15.5%% (ML 21/60)   p62: 21.5%% (ML 27/60)   128e: 0%% (ML 0/60)")
PYEOF

L "=== [4/4] purge merged 62e (keep result json + capture) ==="
rm -rf "$OUT" && L "purged $OUT"
L "RUN_DONE $NAME"
