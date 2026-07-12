#!/usr/bin/env bash
# Generic: build a 62e bf16 variant from a drop map + loop_screen it.
# Usage: build_screen_variant.sh <drop_map> <out_model_dir> <name> <res_dir> [gpu]
set -uo pipefail
DROP="$1"; OUT="$2"; NAME="$3"; RES="$4"; GPU="${5:-0}"
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SCR=/srv/ml/scripts
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
SAMPLE=/mnt/sdc/ml/corpora/loop_screen_sample.jsonl
mkdir -p "$RES"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L(){ echo "[$NAME $(date -u +%H:%M:%S)] $*"; }

L "=== BUILD $NAME (expert_drop, GPU$GPU) ==="
if "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$DROP" --output-dir "$OUT"; then
  L "build OK -> $OUT"
else L "!!! BUILD FAILED"; exit 1; fi

L "=== SCREEN $NAME (200-prompt loop sample, greedy bf16, max-new 2048) ==="
if "$PY" "$SCR/loop_screen.py" --model "$OUT" --out "$RES/loop_$NAME.json" --name "$NAME" \
      --sample "$SAMPLE" --bs 16 --max-new 2048; then
  "$PY" - "$RES/loop_$NAME.json" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1]))
bb=d.get("by_bucket",{})
print("[%s] RESULT loop_pct=%s loops=%s/%s"%(d["name"],d.get("loop_pct"),d.get("loops"),d.get("n")))
print("[%s] by_bucket: "%d["name"]+"  ".join("%s=%s/%s"%(b,v.get("loops"),v.get("n")) for b,v in sorted(bb.items())))
print("[%s] anchors: p62=21.5%% (ML 27/60)  A2+PES=15.5%% (ML 21/60)  format8=17.0%% (ML 26/60)  128e=0%%"%d["name"])
PYEOF
else L "!!! SCREEN FAILED"; exit 1; fi
# disk hygiene: purge the 27G bf16 merge, keep the result json + drop map
rm -rf "$OUT" && L "purged $OUT (kept $RES/loop_$NAME.json)"
L "VARIANT_DONE"
