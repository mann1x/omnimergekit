#!/usr/bin/env bash
# T191 closed-form redistribution: capture(driver corpus) -> fit/emit a recovered
# 62e -> loop_screen vs A2 anchors. The capture corpus MUST be disjoint from the
# loop_screen sample (or the fold is fit on the test set).
#
# Usage: redist_run.sh [--run] <driver> <method> <calib_corpus> [gpu] [max_seqs]
#   Dry-run by default (prints the plan, exits 0). Pass --run to execute.
#   Config: copy redist_config.sh.example -> redist_config.sh (same dir), or export
#   REDIST_* in the environment. See docs/METHOD_redist_framework.md.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
[ -f "$SCRIPT_DIR/redist_config.sh" ] && . "$SCRIPT_DIR/redist_config.sh"
DO_RUN=0; ARGS=()
for a in "$@"; do [ "$a" = "--run" ] && DO_RUN=1 || ARGS+=("$a"); done
set -- "${ARGS[@]+"${ARGS[@]}"}"

DRIVER="${1:-multilingual}"; METHOD="${2:-ream}"; CORPUS="${3:-}"
GPU="${4:-0}"; MAXSEQ="${5:-120}"
REDIST_PY="${REDIST_PY:-python}"
REDIST_SCRIPTS_DIR="${REDIST_SCRIPTS_DIR:-$REPO_ROOT/scripts}"
WORK="${REDIST_WORK:-$PWD/redist_work}"
RES="${REDIST_RESULTS:-$PWD/eval_results_redist}"
OUT_BASE="${REDIST_OUT_BASE:-$PWD/redist_models}"
NAME="${DRIVER}_${METHOD}"
OUT="$OUT_BASE/redist_${NAME}_62e"

cat <<PLAN
=== redist_run plan (driver=$DRIVER method=$METHOD) ===
  python        : $REDIST_PY
  scripts-dir   : $REDIST_SCRIPTS_DIR
  teacher       : ${REDIST_TEACHER:-<unset REDIST_TEACHER>}
  student       : ${REDIST_STUDENT:-<unset REDIST_STUDENT>}
  keep-meta     : ${REDIST_KEEP_META:-<unset REDIST_KEEP_META>}
  sample        : ${REDIST_SAMPLE:-<unset REDIST_SAMPLE>}
  calib corpus  : ${CORPUS:-<arg 3 missing>}
  workdir       : $WORK
  results       : $RES
  emit 62e ->   : $OUT   (purged after screen)
  gpu / max_seqs: $GPU / $MAXSEQ
PLAN
[ "$DO_RUN" = 1 ] || { echo "(dry-run: pass --run to execute)"; exit 0; }

: "${REDIST_TEACHER:?set REDIST_TEACHER (128e teacher dir)}"
: "${REDIST_STUDENT:?set REDIST_STUDENT (pruned 62e dir)}"
: "${REDIST_KEEP_META:?set REDIST_KEEP_META (a2 keep metadata json)}"
: "${REDIST_SAMPLE:?set REDIST_SAMPLE (loop_screen jsonl)}"
[ -n "$CORPUS" ] || { echo "FAIL: arg 3 <calib_corpus> required"; exit 1; }
mkdir -p "$RES" "$WORK" "$OUT_BASE"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WORK/logs"; exec > >(tee -a "$WORK/logs/redist_run_${NAME}.log") 2>&1
L(){ echo "[$NAME $(date -u +%H:%M:%S)] $*"; }

L "=== [1/4] capture (driver=$DRIVER corpus=$(basename "$CORPUS") seqs=$MAXSEQ) ==="
"$REDIST_PY" "$REDIST_SCRIPTS_DIR/redist.py" capture --driver "$DRIVER" --method "$METHOD" \
  --teacher "$REDIST_TEACHER" --corpus "$CORPUS" --keep-meta "$REDIST_KEEP_META" \
  --max-seqs "$MAXSEQ" --max-tokens 512 --device cuda:0 --workdir "$WORK" \
  --scripts-dir "$REDIST_SCRIPTS_DIR" || { L "CAPTURE FAIL"; exit 1; }
CAP="$WORK/capture_${DRIVER}_${METHOD}.pt"

L "=== [2/4] fit + emit ($METHOD -> recovered 62e) ==="
rm -rf "$OUT"
"$REDIST_PY" "$REDIST_SCRIPTS_DIR/redist.py" redistribute --method "$METHOD" --driver "$DRIVER" \
  --capture "$CAP" --teacher "$REDIST_TEACHER" --student "$REDIST_STUDENT" \
  --keep-meta "$REDIST_KEEP_META" --emit "$OUT" --device cuda:0 \
  --scripts-dir "$REDIST_SCRIPTS_DIR" || { L "FIT/EMIT FAIL"; exit 1; }

L "=== [3/4] loop_screen (200-prompt, greedy bf16, max-new 2048) ==="
"$REDIST_PY" "$REDIST_SCRIPTS_DIR/loop_screen.py" --model "$OUT" --out "$RES/loop_$NAME.json" \
  --name "$NAME" --sample "$REDIST_SAMPLE" --bs 16 --max-new 2048 || { L "SCREEN FAIL"; exit 1; }
"$REDIST_PY" - "$RES/loop_$NAME.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); bb = d.get("by_bucket", {})
print("[%s] loop_pct=%s  loops=%s/%s" % (d["name"], d.get("loop_pct"), d.get("loops"), d.get("n")))
print("  by_bucket: " + "  ".join("%s=%s/%s" % (b, v.get("loops"), v.get("n")) for b, v in sorted(bb.items())))
print("  ANCHORS  A2+PES: 15.5%% (ML 21/60)   p62: 21.5%% (ML 27/60)   128e: 0%% (ML 0/60)")
PYEOF

L "=== [4/4] purge merged 62e (keep result json + capture) ==="
rm -rf "$OUT" && L "purged $OUT"
L "RUN_DONE $NAME"
