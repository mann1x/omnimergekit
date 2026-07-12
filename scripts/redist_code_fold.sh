#!/usr/bin/env bash
# T193 code-driver redistribution: capture(code corpus) -> closed-form fold of dropped
# code-experts into A2 fixed survivors -> loop_screen(regression) -> KEEP model for HE+/MPE eval.
# Code fails FLUENTLY, so loop_screen is a no-regression gate; HE+/MPE pass@1 is the primary metric (run separately).
# Closed-form (REAM) = no label leakage despite prompt overlap with eval bench.
# Usage: redist_code_fold.sh <method> <gpu> [max_seqs] [max_tokens]
set -uo pipefail
METHOD="${1:-ream}"; GPU="${2:-0}"; MAXSEQ="${3:-200}"; MAXTOK="${4:-512}"
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SCR=/srv/ml/scripts
TEACHER=/srv/ml/models/base/gemma-4-26B-A4B-it
STUDENT=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it
KM=/srv/ml/scripts/a2_keep_metadata.json
CORPUS=/mnt/sdc/ml/corpora/redist_calib_code.jsonl
SAMPLE=/mnt/sdc/ml/corpora/loop_screen_sample.jsonl
WORK=/srv/ml/redist_work
RES=/srv/ml/eval_results_redist
OUT=/mnt/sdc/ml/google/redist_code_${METHOD}_62e
mkdir -p "$RES" "$WORK"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L(){ echo "[code_$METHOD $(date -u +%H:%M:%S)] $*"; }
CAP="$WORK/capture_code_${METHOD}.pt"

L "=== [1/3] capture (driver=code corpus=redist_calib_code seqs=$MAXSEQ tok=$MAXTOK) ==="
"$PY" "$SCR/redist.py" capture --driver code --method "$METHOD" --teacher "$TEACHER" \
  --corpus "$CORPUS" --keep-meta "$KM" --max-seqs "$MAXSEQ" --max-tokens "$MAXTOK" \
  --device cuda:0 --workdir "$WORK" --scripts-dir "$SCR" || { L "CAPTURE FAIL"; exit 1; }

L "=== [2/3] $METHOD fit + emit (dropped code-experts -> A2 survivors, KEEP) ==="
rm -rf "$OUT"
"$PY" "$SCR/redist.py" redistribute --method "$METHOD" --driver code --capture "$CAP" \
  --teacher "$TEACHER" --student "$STUDENT" --keep-meta "$KM" --emit "$OUT" \
  --device cuda:0 --scripts-dir "$SCR" || { L "FIT/EMIT FAIL"; exit 1; }

L "=== [3/3] loop_screen REGRESSION check (200-prompt, greedy, max-new 2048) ==="
"$PY" "$SCR/loop_screen.py" --model "$OUT" --out "$RES/loop_code_${METHOD}.json" --name "code_${METHOD}" \
  --sample "$SAMPLE" --bs 16 --max-new 2048 || { L "SCREEN FAIL"; exit 1; }
"$PY" - "$RES/loop_code_${METHOD}.json" <<PYEOF
import json
d=json.load(open("$RES/loop_code_${METHOD}.json")); bb=d.get("by_bucket",{})
print("[%s] loop_pct=%s loops=%s/%s" % (d["name"], d.get("loop_pct"), d.get("loops"), d.get("n")))
print("  by_bucket: "+"  ".join("%s=%s/%s"%(b,v.get("loops"),v.get("n")) for b,v in sorted(bb.items())))
print("  A2 ANCHOR 15.5%% (constr 6 ML 21 oe 2)  -> regression gate: must not WORSEN")
PYEOF
L "FOLD_DONE method=$METHOD out=$OUT  (model KEPT for HE+/MPE eval)"
