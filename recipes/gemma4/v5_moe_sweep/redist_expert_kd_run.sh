#!/usr/bin/env bash
# T192 E-ExpertKD runner: KD-train A2's trainable survivor experts to the 128e teacher's
# next-token distribution (logit forward-KL), then loop_screen vs A2 anchors.
# NO --corpus-pad-c4 (English C4 would dilute the multilingual signal); --epochs 100
# cycles the small targeted corpus under the --max-steps cap instead.
#
# Usage: redist_expert_kd_run.sh [--run] <name> <corpus> <train_tensors> <train_layers> \
#                                <steps> <lr> [gpu] [teacher_load] [seqlen]
#   Dry-run by default (prints the plan, exits 0). Pass --run to execute.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
[ -f "$SCRIPT_DIR/redist_config.sh" ] && . "$SCRIPT_DIR/redist_config.sh"
DO_RUN=0; ARGS=()
for a in "$@"; do [ "$a" = "--run" ] && DO_RUN=1 || ARGS+=("$a"); done
set -- "${ARGS[@]+"${ARGS[@]}"}"

NAME="${1:-}"; CORPUS="${2:-}"; TT="${3:-experts+router}"; TL="${4:-mid}"
STEPS="${5:-200}"; LR="${6:-2e-5}"; GPU="${7:-0}"; TLOAD="${8:-4bit}"; SEQ="${9:-1024}"
REDIST_PY="${REDIST_PY:-python}"
REDIST_SCRIPTS_DIR="${REDIST_SCRIPTS_DIR:-$REPO_ROOT/scripts}"
WORK="${REDIST_WORK:-$PWD/redist_work}"
RES="${REDIST_RESULTS:-$PWD/eval_results_redist}"
OUT_BASE="${REDIST_OUT_BASE:-$PWD/redist_models}"
OUT="$OUT_BASE/ekd_${NAME}_62e"

cat <<PLAN
=== redist_expert_kd_run plan (name=$NAME) ===
  python      : $REDIST_PY
  scripts-dir : $REDIST_SCRIPTS_DIR
  teacher     : ${REDIST_TEACHER:-<unset REDIST_TEACHER>}
  student     : ${REDIST_STUDENT:-<unset REDIST_STUDENT>}
  sample      : ${REDIST_SAMPLE:-<unset REDIST_SAMPLE>}
  corpus      : ${CORPUS:-<arg 2 missing>}
  tt/layers   : $TT / $TL    steps=$STEPS lr=$LR teacher=$TLOAD seq=$SEQ
  out ->      : $OUT   (purged after screen)
  gpu         : $GPU
PLAN
[ "$DO_RUN" = 1 ] || { echo "(dry-run: pass --run to execute)"; exit 0; }

[ -n "$NAME" ]   || { echo "FAIL: arg 1 <name> required"; exit 1; }
[ -n "$CORPUS" ] || { echo "FAIL: arg 2 <corpus> required"; exit 1; }
: "${REDIST_TEACHER:?set REDIST_TEACHER (128e teacher dir)}"
: "${REDIST_STUDENT:?set REDIST_STUDENT (pruned 62e dir)}"
: "${REDIST_SAMPLE:?set REDIST_SAMPLE (loop_screen jsonl)}"
mkdir -p "$RES" "$WORK" "$WORK/logs" "$OUT_BASE"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec > >(tee -a "$WORK/logs/redist_ekd_${NAME}.log") 2>&1
L(){ echo "[ekd:$NAME $(date -u +%H:%M:%S)] $*"; }

L "=== [1/3] E-ExpertKD train (tt=$TT layers=$TL steps=$STEPS lr=$LR teacher=$TLOAD seq=$SEQ corpus=$(basename "$CORPUS")) ==="
rm -rf "$OUT"
"$REDIST_PY" "$REDIST_SCRIPTS_DIR/router_kd.py" \
  --base-dir "$REDIST_TEACHER" --variant-dir "$REDIST_STUDENT" --out-dir "$OUT" \
  --train-tensors "$TT" --train-layers "$TL" \
  --student-load bf16 --teacher-load "$TLOAD" \
  --teacher-device '{"":0}' --student-device '{"":0}' --gpu-mem-gib 85 \
  --optim paged_adamw8bit --grad-checkpointing \
  --corpus-file "$CORPUS" --epochs 100 --max-samples 100000 \
  --tau 1.0 --lr "$LR" --max-steps "$STEPS" --batch-size 1 --grad-accum 8 \
  --max-seq-len "$SEQ" --no-canary --log-every 10 || { L "KD FAIL"; exit 1; }

L "=== [2/3] loop_screen (200-prompt, greedy bf16, max-new 2048) ==="
"$REDIST_PY" "$REDIST_SCRIPTS_DIR/loop_screen.py" --model "$OUT" --out "$RES/loop_ekd_$NAME.json" \
  --name "ekd_$NAME" --sample "$REDIST_SAMPLE" --bs 16 --max-new 2048 || { L "SCREEN FAIL"; exit 1; }
"$REDIST_PY" - "$RES/loop_ekd_$NAME.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); bb = d.get("by_bucket", {})
print("[%s] loop_pct=%s  loops=%s/%s" % (d["name"], d.get("loop_pct"), d.get("loops"), d.get("n")))
print("  by_bucket: " + "  ".join("%s=%s/%s" % (b, v.get("loops"), v.get("n")) for b, v in sorted(bb.items())))
print("  ANCHORS  A2+PES 15.5%% (ML 21/60)   p62 21.5%% (ML 27/60)   128e 0%% (ML 0/60)")
PYEOF

L "=== [3/3] purge merged model (keep result json) ==="
rm -rf "$OUT" && L "purged $OUT"
L "EKD_DONE $NAME"
