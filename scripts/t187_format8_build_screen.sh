#!/usr/bin/env bash
# T187: build fc15_25-p8-format8 (force-keep 8 diagnostic format experts) + loop_screen.
#
# CLEAN ISOLATION test of the council RUN-IT hypothesis. The drop map differs from
# A2 by EXACTLY 8 experts (force-keep L0 e125 / L1 e47 / L9 e66 / L12 e80 /
# L15 e2 / L15 e67 / L16 e126 / L17 e53; evict the 8 lowest-agg survivors in those
# layers). Everything else identical to A2 — unlike T176 which swapped 696/1980.
#
# Comparison anchors (same loop_screen.py + same 200-prompt sample):
#   p62 pristine (A2 keep-set, NO PES) = 21.5%   <- format8 is also pristine: APPLES-TO-APPLES
#   A2 = fc15_25-p8 + PES120           = 15.5%
#   128e unpruned                      =  0.0%
# Gate: format8 << 21.5% (esp. multilingual/constrained buckets) => format experts
#       are causal => apply PES + re-screen for deployable number. ~21.5% => FALSIFY.
set -uo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SCR=/srv/ml/scripts
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
DROP=/srv/ml/scripts/format8_62e_fc15_25_p8_drop_map.json
OUT=/mnt/sdc/ml/google/gemma-4-A4B-62e-format8-it
SAMPLE=/mnt/sdc/ml/corpora/loop_screen_sample.jsonl
RES=/srv/ml/eval_results_tracks_2_3/t187_format8
mkdir -p "$RES"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L(){ echo "[t187 $(date -u +%H:%M:%S)] $*"; }

L "=== BUILD format8 62e bf16 (expert_drop) ==="
if "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$DROP" --output-dir "$OUT"; then
  L "build OK -> $OUT"
else
  L "!!! BUILD FAILED"; exit 1
fi

L "=== SCREEN format8 (200-prompt loop sample, greedy bf16, max-new 2048) ==="
if "$PY" "$SCR/loop_screen.py" --model "$OUT" --out "$RES/loop_format8.json" --name format8 \
      --sample "$SAMPLE" --bs 16 --max-new 2048; then
  "$PY" - "$RES/loop_format8.json" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1]))
print("[t187] RESULT format8 loop_pct=%s loops=%s/%s"%(d.get("loop_pct"),d.get("loops"),d.get("n")))
bb=d.get("by_bucket",{})
print("[t187] by_bucket: "+"  ".join("%s=%s/%s"%(b,v.get("loops"),v.get("n")) for b,v in sorted(bb.items())))
print("[t187] anchors: p62-pristine=21.5%  A2(+PES)=15.5%  128e=0.0%")
PYEOF
else
  L "!!! SCREEN FAILED"; exit 1
fi
L "T187_DONE"
